import os
import sys
# Get the absolute path to the parent directory (assumes this file is in 'condensation')
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
# autoreload modules
""" %load_ext autoreload
%autoreload 2 """
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
import pandas as pd
from chatbot_api.providers.openai import OpenAIProvider
from chatbot_api.base import Role, Message
from .argument import Argument
from .parser import OutputParser

class Condenser:
    """Condenses source comments into distinct arguments using an LLM."""
    
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.parser = OutputParser()
        self.all_arguments: List[Argument] = []
        
        self.PROMPT_TEMPLATE = """
        ### Ohjeet:
        1. Käy läpi seuraavat kommentit aiheesta: "{topic}"
        2. Jos kommenteissa on näkökulmia, joita ei vielä ole olemassa olevissa argumenteissa:
           - Luo uusi argumentti jokaiselle uudelle näkökulmalle
           - Varmista, että uusi argumentti on selkeästi erilainen kuin olemassa olevat
           - Kirjoita argumentti yhdellä virkkeellä
        3. Huomioi:
           - Älä muokkaa olemassa olevia argumentteja
           - Luo uusi argumentti vain, jos se tuo esiin täysin uuden näkökulman
           - Merkitse selkeästi, mitkä kommentit (numerot) liittyvät kuhunkin uuteen argumenttiin
           - Käytä kahta parhaiten sopivaa kommenttia uuden argumentin lähteenä

        ### Olemassa olevat argumentit:
        {existing_arguments}

        ### Uudet kommentit:
        {comments}

        ### Tulostusmuoto:
        <ARGUMENTS>
        ARGUMENTTI 1: [Uusi argumentti, jos löytyy uusi näkökulma]
        Lähteet: [Kommenttien numerot]
        </ARGUMENTS>
        """

    async def process_comments(self, comments: List[str], topic: str, n_comments: int, batch_size: int = 10) -> List[Argument]:
        """Process all comments in batches to generate arguments."""
        n_iterations = n_comments // batch_size
        for i in range(n_iterations):
            print(f"Processing batch {i+1} of {n_iterations}...")
            batch = comments[i:i + batch_size]
            new_args = await self._process_batch(batch, self.all_arguments, topic, i)
            self.all_arguments.extend(new_args)
            
        return self.all_arguments

    async def _process_batch(self, 
                           batch: List[str], 
                           existing_args: List[Argument], 
                           topic: str,
                           start_idx: int) -> List[Argument]:
        """Process a single batch of comments."""
        # Format comments with indices
        comments_text = "\n".join(f"Kommentti {i+1}: {comment}" 
                                for i, comment in enumerate(batch))
        
        # Format existing arguments
        existing_args_text = "\n".join(f"Argumentti {i+1}: {arg.main_argument}" 
                                     for i, arg in enumerate(existing_args))
        
        # Create prompt
        prompt = self.PROMPT_TEMPLATE.format(
            topic=topic,
            existing_arguments=existing_args_text if existing_args else "Ei olemassa olevia argumentteja.",
            comments=comments_text
        )
        
        # Get response from LLM
        response = await self.llm_provider.generate([Message(Role.USER, prompt)])
        
        # Parse new arguments and their sources
        new_arg_texts = self.parser.parse_arguments(response.content)
        source_indices_per_arg = self.parser.parse_source_indices(response.content)
        
        # Create new Argument objects
        new_arguments = []
        for i, arg_text in enumerate(new_arg_texts):
            # Get source indices for this specific argument
            local_indices = source_indices_per_arg[i] if i < len(source_indices_per_arg) else []
            
            # Convert batch-relative indices to global indices
            global_indices = [start_idx * len(batch) + (idx - 1) for idx in local_indices]
            
            # Get the actual source comments using local indices
            sources = [batch[idx-1] for idx in local_indices if 0 <= idx-1 < len(batch)]
            
            new_arguments.append(Argument(
                main_argument=arg_text,
                sources=sources,
                source_indices=global_indices,
                topic=topic
            ))
            
        return new_arguments


