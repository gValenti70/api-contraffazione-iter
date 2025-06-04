from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import AzureOpenAI
import os
import json
import logging

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_APIKEY"),
    api_version="2024-12-01-preview",
    azure_endpoint="https://openaifashion.openai.azure.com/"
)

app = FastAPI()

deployment = "gpt-4o"

# Modello dati in input
class OggettoInput(BaseModel):
    tipologia: Optional[str] = "borsa"
    immagini: List[str]

@app.post("/analizza-oggetto")
async def analizza_oggetto(input: OggettoInput):
    tipologia = input.tipologia or "borsa"
    num_foto = len(input.immagini)

    if num_foto == 0:
        raise HTTPException(status_code=400, detail="Nessuna immagine inviata.")

    if num_foto == 1:
        prompt = (
            f"Stai analizzando un oggetto che sembra essere una '{tipologia}'. "
            f"Ti è stata fornita una sola immagine.\n\n"
            f"Analizza i dettagli visivi per determinare se sembra un oggetto autentico o contraffatto. "
            f"Riconosci anche la marca visibile o probabile dell'oggetto, se presente, e usala nel tuo ragionamento.\n\n"
            f"Se la foto non mostra l'oggetto chiaramente, restituisci:\n"
            f"- \"percentuale\": -1\n"
            f"- \"motivazione\": spiega perché non è valutabile\n\n"
            f"> Altrimenti, indica quale zona andrebbe fotografata meglio per una valutazione più affidabile. "
            f"Stima la probabilità di contraffazione seguendo queste linee guida:\n"
            f"- Oggetto apparentemente autentico -> 0-20\n"
            f"- Dettagli plausibili ma serve conferma -> 20-40\n"
            f"- Elementi sospetti -> 40-70\n"
            f"- Forti segnali di falsificazione -> 70-100\n\n"
            f"La percentuale deve essere coerente con la tua motivazione.\n\n"
            f"Rispondi solo in JSON:\n"
            "{\n"
            "  \"percentuale\": numero intero (0–100 oppure -1),\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"richiedi_altra_foto\": true,\n"
            "  \"dettaglio_richiesto\": \"stringa\"\n"
            "}"
        )

    elif num_foto == 2:
        prompt = (
            f"Stai analizzando un oggetto che sembra essere una '{tipologia}', basandoti su due immagini.\n\n"
            f"Analizza i dettagli visivi per determinare se sembra autentico o contraffatto. "
            f"Riconosci autonomamente la marca, se possibile.\n\n"
            f"Se le immagini non sono pertinenti, restituisci -1 con spiegazione.\n\n"
            f"Stima la probabilità secondo questi intervalli:\n"
            f"- Dettagli autentici -> 0-20\n"
            f"- Dettagli plausibili ma non confermati -> 20-40\n"
            f"- Alcuni segnali anomali -> 40-70\n"
            f"- Forti sospetti -> 70-100\n\n"
            f"La motivazione deve spiegare la percentuale.\n"
            f"Se servono altri dettagli, indicane uno da fotografare.\n\n"
            f"Rispondi solo in JSON:\n"
            "{\n"
            "  \"percentuale\": numero intero (0–100 oppure -1),\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"richiedi_altra_foto\": true o false,\n"
            "  \"dettaglio_richiesto\": \"stringa\"\n"
            "}"
        )

    else:
        prompt = (
            f"Stai analizzando un oggetto che sembra essere una '{tipologia}', usando tre fotografie.\n\n"
            f"Analizza i dettagli per determinare l'autenticità dell'oggetto. "
            f"Riconosci autonomamente la marca dall'aspetto e dai segni distintivi.\n\n"
            f"Se nessuna immagine è pertinente, restituisci -1.\n\n"
            f"Altrimenti, usa questi criteri:\n"
            f"- Oggetto autentico in ogni dettaglio -> 0-20\n"
            f"- Coerente ma serve conferma -> 20-40\n"
            f"- Dettagli sospetti o incoerenti -> 40-70\n"
            f"- Chiari segni di falsificazione -> 70-100\n\n"
            f"La percentuale deve essere coerente con la motivazione.\n\n"
            f"Rispondi solo in JSON:\n"
            "{\n"
            "  \"percentuale\": numero intero (0–100 oppure -1),\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"richiedi_altra_foto\": false,\n"
            "  \"dettaglio_richiesto\": \"\"\n"
            "}"
        )

    messaggi = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                    for img in input.immagini
                ]
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messaggi
        )
        contenuto = response.choices[0].message.content.strip()

        if contenuto.startswith("```"):
            contenuto = contenuto.split("```")[1].strip()
            if contenuto.startswith("json"):
                contenuto = contenuto[4:].strip()

        json_output = json.loads(contenuto)

        campi_attesi = ["percentuale", "motivazione", "richiedi_altra_foto", "dettaglio_richiesto"]
        for campo in campi_attesi:
            if campo not in json_output:
                raise HTTPException(status_code=500, detail=f"Campo mancante: {campo}")

        return json_output

    except Exception as e:
        logging.error(f"Errore backend: {e}")
        raise HTTPException(status_code=500, detail="Errore durante l'elaborazione della richiesta.")

