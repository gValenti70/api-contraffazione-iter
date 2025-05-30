from fastapi import FastAPI
from pydantic import BaseModel
from openai import AzureOpenAI
from typing import List
import os
import json

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

    # Prompt dinamico
    if num_foto == 1:
        prompt = (
            f"Stai analizzando un oggetto di tipo '{input.tipologia}' della marca '{input.marca}'. "
            f"Ti è stata fornita solo una fotografia. In base a questa immagine, indica quale dettaglio chiave dovrebbe essere fotografato in modo ravvicinato "
            f"per aiutarti a determinare l'autenticità dell'oggetto.\n"
            f"Rispondi in formato JSON:\n"
            "{\n"
            "  \"percentuale\": numero intero,\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"richiedi_altra_foto\": true,\n"
            "  \"dettaglio_richiesto\": \"stringa\"\n"
            "}"
        )
    elif num_foto == 2:
        prompt = (
            f"Stai analizzando un oggetto di tipo '{input.tipologia}' della marca '{input.marca}'. "
            f"Hai ricevuto due fotografie. In base a queste immagini, puoi stimare la probabilità che l'oggetto sia contraffatto. "
            f"Se hai ancora dubbi, suggerisci un dettaglio chiave da fotografare meglio per aiutarti. "
            f"Rispondi in JSON come segue:\n"
            "{\n"
            "  \"percentuale\": numero intero,\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"richiedi_altra_foto\": true/false,\n"
            "  \"dettaglio_richiesto\": \"stringa (vuota se non necessaria)\"\n"
            "}"
        )
    else:
        prompt = (
            f"Hai ricevuto 3 fotografie di un oggetto di tipo '{input.tipologia}' della marca '{input.marca}'. "
            f"Analizzale e valuta se si tratta di un oggetto autentico o contraffatto. "
            f"Fornisci una percentuale di contraffazione stimata e le motivazioni.\n"
            f"Rispondi solo in formato JSON:\n"
            "{\n"
            "  \"percentuale\": numero intero,\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"richiedi_altra_foto\": false,\n"
            "  \"dettaglio_richiesto\": \"\"\n"
            "}"
        )

    # Messaggi
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

        # Sanifica contenuto da blocchi Markdown
        if contenuto.startswith("```"):
            contenuto = contenuto.split("```")[1].strip()
            if contenuto.startswith("json"):
                contenuto = contenuto[4:].strip()

        try:
            json_output = json.loads(contenuto)
        except json.JSONDecodeError as e:
            logging.error(f"Errore JSON: {e}\nRisposta:\n{contenuto}")
            raise HTTPException(status_code=500, detail="⚠️ Errore nel parsing della risposta JSON")

        return json_output

    except Exception as e:
        logging.error(f"Errore generale: {e}")
        raise HTTPException(status_code=500, detail="Errore durante l'elaborazione della richiesta.")
