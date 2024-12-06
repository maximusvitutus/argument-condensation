import os
import sys

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import asyncio
from chatbot_api.providers.openai import OpenAIProvider
from chatbot_api.base import EmbeddingResponse, Message, Role
from dotenv import load_dotenv
from pathlib import Path
from data.data_utils import from_pdf_to_string

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env" # move up one directory to find the .env file
success = load_dotenv(dotenv_path=env_path, override=True)
key = os.getenv("OPENAI_API_KEY")
script_dir = os.path.dirname(os.path.abspath(__file__))

provider = OpenAIProvider(api_key=key)

# Files that are used to prompt the model

parties = {
        "KD": "data/manifestos/kd/kd.txt",
        "KESK": "data/manifestos/kesk/kesk.txt",
        "KOK": "data/manifestos/kok/kok.txt",
        "VIHR": "data/manifestos/vihr/vihr.txt",
        "VAS": "data/manifestos/vas/vas.txt",
        "PS": "data/manifestos/ps/ps.txt",
        "SDP": "data/manifestos/sdp/sdp.txt",
        } 

async def main():
    party = input("""
Tervetuloa juttelemaan puolueille! Tässä sovelluksessa voit puhua chatbotille,
joka vastaa kysymyksiisi puolueen vaaliohjelman perusteella. Muista, että chatissä
esitetyt mielipiteet ovat generoitu tekoälyn avulla. Tarkista vastausten todenmukaisuus
ja anna palautetta epätarkkuuksista!

Valitse näistä puolueista ja kirjoita terminaaliin puolueen tunnus:
KD
KESK
KOK
VIHR
VAS
PS
SDP (manifesto too large for current model)

Lopettaaksesi keskustelun kirjoita "Heippa!"\n
""")
    

    
    # TODO: accept other common spellings
    while party not in parties:  
        party = input("Puolue ei ollut listassa, yritä uudelleen.\n")

    manifesto_path = os.path.join(script_dir,  parties[party])

    manifesto = ""
    with open(manifesto_path, "r") as file:
        manifesto = file.read()

    first_message = f""" Olet chatbotti, joka edustaa suomalaista puoluetta {party}. Vastaa kohteliaasti äläkä 
        kiroilla tai muuten käyttää loukkaavaa kieltä. Puhu Suomea. Älä vastaa provokaatioihin tai
        keskustelukumppanisi provokaatioyrityksiin. Vastaat kysymyksiin, jotka perustuvat arvoihin ja
        tavoitteisiin, jotka on esitetty tässä puoluemanifestissa:
    """

    prompt = first_message + manifesto

    # print(provider.estimated_cost(await provider.count_tokens(prompt), 500.0))
    # cost of sending the manifesto: 0.046865500000000004 (of ? currency)
    
    first_message = [Message(role=Role("system"), content=prompt)]

    response = await provider.generate(
            first_message,
            temperature=0.7,
            )

    print(response.content + "\n")


    user_input = ""
    while user_input != "Heippa!":
        user_input = input("Käyttäjä: ")
        message = [Message(role=Role("user"), content=user_input)]
        response = await provider.generate(
                message,
                temperature=0.7,
                )
        print("\n" + response.content + "\n")

asyncio.run(main())
