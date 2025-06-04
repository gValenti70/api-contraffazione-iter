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
            f"Prima verifica se l'immagine è rilevante: se non mostra chiaramente l'oggetto richiesto, restituisci:\n"
            f"- \"percentuale\": -1\n"
            f"- \"motivazione\": spiega che la foto è fuori contesto.\n\n"
            f"Se è rilevante, spiega cosa manca per determinare l'autenticità e indica un dettaglio da fotografare meglio.\n"
            f"La percentuale deve essere coerente: bassa se rassicurante, alta se sospetta.\n\n"
            f"Rispondi in JSON:\n"
            "{\n"
            "  \"percentuale\": numero intero (0–100 oppure -1),\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"richiedi_altra_foto\": true,\n"
            "  \"dettaglio_richiesto\": \"stringa\"\n"
            "}"
        )
    elif num_foto == 2:
        prompt = (
            f"Stai analizzando un oggetto di tipo '{input.tipologia}' della marca '{input.marca}'. "
            f"Hai ricevuto due fotografie.\n\n"
            f"Se le immagini non sono pertinenti all'oggetto, restituisci -1 come percentuale con una motivazione.\n"
            f"Altrimenti, analizza i dettagli e indica la probabilità di contraffazione in modo coerente con la tua analisi: "
            f"bassa se l'oggetto sembra autentico, alta se sospetto. Se hai bisogno di un'altra immagine, specifica quale.\n\n"
            f"Rispondi in JSON:\n"
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
            f"Se le immagini non mostrano l'oggetto corretto, restituisci -1 come percentuale e spiega.\n"
            f"Se sono pertinenti, analizza attentamente e fornisci una stima della probabilità di contraffazione "
            f"coerente con la tua motivazione: bassa se rassicurante, alta se sospetta.\n\n"
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
