from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
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

class OggettoInput(BaseModel):
    tipologia: str
    marca: str
    immagini: List[str]

@app.post("/analizza-oggetto")
async def analizza_oggetto(input: OggettoInput):
    num_foto = len(input.immagini)

    if num_foto == 0:
        raise HTTPException(status_code=400, detail="Nessuna immagine inviata.")

    if num_foto == 1:
        prompt = (
            f"Stai analizzando un oggetto di tipo '{input.tipologia}' della marca '{input.marca}'. "
            f"Ti è stata fornita solo una fotografia.\n\n"
            f"Se l'immagine non mostra un oggetto coerente con la tipologia richiesta, restituisci:\n"
            f"- \"percentuale\": -1\n"
            f"- \"motivazione\": spiega che la foto non è rilevante\n\n"
            f"> Altrimenti, indica quale parte dell'oggetto andrebbe fotografata meglio per valutarne l'autenticità. "
            f"Fornisci anche una stima della probabilità di contraffazione, secondo queste regole:\n"
            f"- Oggetto apparentemente autentico -> percentuale tra 0 e 20\n"
            f"- Dettagli plausibili ma serve conferma -> 20-40\n"
            f"- Elementi sospetti -> 40-70\n"
            f"- Segnali forti di contraffazione -> 70-100\n\n"
            f"La percentuale deve essere coerente con la tua motivazione.\n\n"
            f"Rispondi solo in JSON:\n"
            "{\n"
            "  \"percentuale\": numero intero (0–100 oppure -1),\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"richiedi_altra_foto\": true,\n"
            "  \"dettaglio_richiesto\": \"stringa descrittiva\"\n"
            "}"
        )
    
    elif num_foto == 2:
        prompt = (
            f"Stai analizzando un oggetto di tipo '{input.tipologia}' della marca '{input.marca}'. "
            f"Hai ricevuto due fotografie.\n\n"
            f"Se nessuna delle due immagini mostra un oggetto coerente, restituisci:\n"
            f"- \"percentuale\": -1\n"
            f"- \"motivazione\": spiega che non è possibile valutare\n\n"
            f"> Altrimenti, valuta la probabilità di contraffazione secondo i seguenti criteri:\n"
            f"- Dettagli autentici -> 0-20\n"
            f"- Qualità buona ma non del tutto confermata -> 20-40\n"
            f"- Alcuni dettagli anomali -> 40-70\n"
            f"- Forti dubbi -> 70-100\n\n"
            f"La motivazione deve essere coerente con la percentuale.\n"
            f"Se serve un'altra immagine, suggerisci il dettaglio da fotografare.\n\n"
            f"Rispondi solo in JSON:\n"
            "{\n"
            "  \"percentuale\": numero intero (0–100 oppure -1),\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"richiedi_altra_foto\": true o false,\n"
            "  \"dettaglio_richiesto\": \"stringa oppure vuota\"\n"
            "}"
        )
    
    else:
        prompt = (
            f"Hai ricevuto 3 fotografie di un oggetto di tipo '{input.tipologia}' della marca '{input.marca}'.\n\n"
            f"Se nessuna immagine mostra l'oggetto richiesto, restituisci:\n"
            f"- \"percentuale\": -1\n"
            f"- \"motivazione\": spiega che l'oggetto non è riconoscibile o fuori contesto\n\n"
            f"> Altrimenti, valuta il rischio di contraffazione secondo queste linee guida:\n"
            f"- Oggetto conforme in ogni dettaglio -> 0-20\n"
            f"- Oggetto plausibile ma serve conferma -> 20-40\n"
            f"- Dettagli sospetti o incoerenti -> 40-70\n"
            f"- Evidenti segni di falsificazione -> 70-100\n\n"
            f"La percentuale deve essere coerente con la motivazione, che spiega chiaramente il motivo della valutazione.\n\n"
            f"Rispondi solo in formato JSON:\n"
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

        campi = ["percentuale", "motivazione", "richiedi_altra_foto", "dettaglio_richiesto"]
        for campo in campi:
            if campo not in json_output:
                raise HTTPException(status_code=500, detail=f"⚠️ Campo mancante: {campo}")

        return json_output

    except Exception as e:
        logging.error(f"Errore back-end: {e}")
        raise HTTPException(status_code=500, detail="Errore durante l'elaborazione della richiesta.")
