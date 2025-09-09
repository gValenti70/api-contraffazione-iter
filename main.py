from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import AzureOpenAI
import os
import json
import logging

# client = AzureOpenAI(
#     api_key=os.getenv("OPENAI_APIKEY"),
#     api_version="2024-12-01-preview",
#     azure_endpoint="https://apitestfashion.openai.azure.com/"
# )

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_APIKEY"),
    api_version="2024-12-01-preview",
    azure_endpoint="https://contraffazione-fashion-openai.openai.azure.com/"
)
deployment = "gpt-4.1"
app = FastAPI()
# deployment = "gpt-4o"

class OggettoInput(BaseModel):
    tipologia: Optional[str] = "borsa"
    immagini: List[str]

@app.post("/analizza-oggetto")
async def analizza_oggetto(input: OggettoInput):
    tipologia = input.tipologia or "borsa"
    num_foto = len(input.immagini)

    if num_foto == 0:
        raise HTTPException(status_code=400, detail="Nessuna immagine inviata.")

    # Costruzione del prompt in base al numero di immagini
    if num_foto == 1:
        prompt = (
            f"Stai analizzando un oggetto che sembra essere una '{tipologia}'. "
            f"Ti è stata fornita una sola immagine.\n\n"
            f"Analizza i dettagli visivi per determinare se sembra un oggetto autentico o contraffatto. "
            f"Riconosci anche la marca visibile o probabile dell'oggetto e indicala nel campo 'marca_stimata'.\n\n"
            f"Se la foto non mostra l'oggetto chiaramente, restituisci:\n"
            f"- \"percentuale\": -1\n"
            f"- \"motivazione\": spiegazione\n"
            f"- \"marca_stimata\": \"\"\n\n"
            f"Altrimenti, fornisci:\n"
            f"- percentuale stimata tra:\n"
            f"  - 0–20 -> oggetto apparentemente autentico\n"
            f"  - 20–40 -> serve conferma\n"
            f"  - 40–70 -> elementi sospetti\n"
            f"  - 70–100 -> probabile contraffazione\n\n"
            f"Rispondi solo in JSON:\n"
            "{\n"
            "  \"percentuale\": numero intero (0–100 oppure -1),\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"marca_stimata\": \"stringa (vuota se non determinabile)\",\n"
            "  \"richiedi_altra_foto\": true,\n"
            "  \"dettaglio_richiesto\": \"stringa\"\n"
            "}"
        )

    elif num_foto == 2:
        prompt = (
            f"Stai analizzando un oggetto che sembra essere una '{tipologia}', basandoti su due immagini.\n\n"
            f"Analizza i dettagli visivi per stimare l'autenticità e indica la marca riconoscibile nel campo 'marca_stimata'.\n\n"
            f"Se non riconoscibile, lascia marca_stimata vuota. Se le immagini non sono pertinenti, restituisci -1.\n\n"
            f"Classifica la probabilità così:\n"
            f"- 0–20 -> autentico\n"
            f"- 20–40 -> buono ma non sicuro\n"
            f"- 40–70 -> sospetto\n"
            f"- 70–100 -> fortemente falso\n\n"
            f"Rispondi solo in JSON:\n"
            "{\n"
            "  \"percentuale\": numero intero (0–100 oppure -1),\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"marca_stimata\": \"stringa (vuota se non nota)\",\n"
            "  \"richiedi_altra_foto\": true o false,\n"
            "  \"dettaglio_richiesto\": \"stringa\"\n"
            "}"
        )

    else:
        prompt = (
            f"Hai ricevuto 3 immagini di un oggetto che sembra essere una '{tipologia}'. "
            f"Analizza i dettagli e indica se sembra autentico o contraffatto. Riconosci la marca visivamente, e riportala in 'marca_stimata'.\n\n"
            f"Se le immagini non sono pertinenti, restituisci -1.\n\n"
            f"Classifica:\n"
            f"- 0–20 -> autentico\n"
            f"- 20–40 -> coerente ma incerto\n"
            f"- 40–70 -> sospetto\n"
            f"- 70–100 -> molto probabilmente falso\n\n"
            f"Rispondi solo in JSON:\n"
            "{\n"
            "  \"percentuale\": numero intero (0–100 oppure -1),\n"
            "  \"motivazione\": \"stringa\",\n"
            "  \"marca_stimata\": \"stringa (vuota se non nota)\",\n"
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

        campi_attesi = ["percentuale", "motivazione", "richiedi_altra_foto", "dettaglio_richiesto", "marca_stimata"]
        for campo in campi_attesi:
            if campo not in json_output:
                raise HTTPException(status_code=500, detail=f"Campo mancante: {campo}")

        # Se siamo alla terza immagine, forziamo il comportamento finale
        if num_foto >= 3:
            json_output["richiedi_altra_foto"] = False
            json_output["dettaglio_richiesto"] = ""
        
        return json_output


    except Exception as e:
        logging.error(f"Errore backend: {e}")
        raise HTTPException(status_code=500, detail="Errore durante l'elaborazione della richiesta.")

    except Exception as e:
        logging.error(f"Errore backend: {e}")
        raise HTTPException(status_code=500, detail="Errore durante l'elaborazione della richiesta.")

