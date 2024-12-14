import os
import sys
import pandas as pd

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
sys.path.insert(0, parent_dir)

from dataclasses import dataclass
from typing import List
from chatbot_api.providers.openai import OpenAIProvider
from chatbot_api.base import Role, Message

@dataclass
class Argument:
    main_point: str
    subpoints: List[str]
    score_distribution: List[float]
    source_indices: List[int]

class ArgumentProcessor:
    def __init__(self, llm_provider: OpenAIProvider, batch_size):
        self.llm_provider = llm_provider
        self.batch_size = batch_size

    async def _create_initial_prompt(self, comments: List[str], topic: str) -> str:
        return f"""
            ### Instructions:
            1. Review the following comments on the topic: "{topic}"
            2. **Identify and categorize arguments into main topics**.
            - Main arguments are broad, high-level points.
            - Subpoints provide supporting details or reasoning for the main argument.
            3. Avoid redundancy by combining similar arguments or subpoints.
            4. Format your response like this:
            - MAIN: [Main Argument]
            - SUB: [Supporting subpoint]
            5. Use concise language, ensuring each Main Argument stands out on its own.
            6. Provide the output in English, even if the input text is not.
            7. Ensure that no argument is repeated, and that each is categorized correctly.

            ### Output format:  
            <ARGUMENTS>
            MAIN: [Main Argument]
            SUB: [Supporting subpoint]
            MAIN: [Main Argument 2]
            SUB: [Supporting subpoint]
            </ARGUMENTS>

            ### Comments to analyze:
            {' '.join(comments)}"""
    
    async def _create_redundancy_prompt(self, arguments: List[Argument]) -> str:
        # Convert arguments to a formatted string for the prompt
        formatted_args = self._format_arguments_for_prompt(arguments)
        
        return f"""You are an expert at analyzing and organizing complex arguments into clear hierarchical structures.

            ### Task
            Analyze and reorganize these arguments to create a more coherent and non-redundant structure. Focus on:
            1. Properly grouping related ideas
            2. Eliminating redundancy while preserving unique insights
            3. Ensuring logical hierarchy between main points and supporting details

            ### Current Structure Problems
            The current structure has these issues:
            - Some subpoints don't directly support their main arguments
            - Related arguments are split across multiple main points
            - Some ideas are repeated with slightly different wording
            - Main arguments sometimes overlap in scope
            
            ### Guidelines for Reorganization
            1. Main Arguments should:
                - Be broad enough to encompass multiple related subpoints
                - Be mutually exclusive with other main arguments
                - Capture a distinct perspective or theme
                - Use clear, representative language
            
            2. Subpoints should:
                - Directly support or elaborate on their main argument
                - Provide specific evidence, examples, or reasoning
                - Not repeat ideas found in other subpoints
                - Be placed under the most relevant main argument
            
            3. Hierarchy Rules:
                - If two main arguments are making similar points, combine them
                - If a subpoint could support multiple main arguments, place it under the most specific relevant one
                - If a subpoint is actually a main argument, promote it
                - If a main argument is actually a specific example, demote it to a subpoint
                - Ensure that the subpoints are distinct and do not repeat ideas from other subpoints.
                - Merge similar subpoints that express the same idea in a single point.

            ### Input Arguments:
            {formatted_args}

            ### Instructions
            1. First, identify the core themes in the main arguments
            2. Group related main arguments together
            3. Reorganize subpoints under the most relevant main arguments
            4. Eliminate redundancies by combining similar points
            5. Ensure each subpoint provides **unique** support for its main argument

            Maintain the exact format:
            <ARGUMENTS>
            MAIN: [Broad, distinct argument]
            SUB: [Specific supporting point]
            SUB: [Another relevant supporting point]
            </ARGUMENTS>"""
    
    def _format_arguments_for_prompt(self, arguments: List[Argument]) -> str:
        formatted = "<ARGUMENTS>\n"
        for arg in arguments:
            formatted += f"MAIN: {arg.main_point}\n"
            for sub in arg.subpoints:
                formatted += f"SUB: {sub}\n"
        formatted += "</ARGUMENTS>"
        return formatted

    async def _parse_argument_structure(self, text: str) -> List[Argument]:
        arguments = []
        current_main = None
        current_subs = []
        
        try:
            # Extract content between <ARGUMENTS> tags
            content = text.split("<ARGUMENTS>")[1].split("</ARGUMENTS>")[0].strip()
            
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("MAIN:"):
                    if current_main is not None:
                        arguments.append(Argument(current_main, current_subs))
                    current_main = line[5:].strip()
                    current_subs = []
                elif line.startswith("SUB:"):
                    current_subs.append(line[4:].strip())
            
            if current_main is not None:
                arguments.append(Argument(current_main, current_subs))
                
        except Exception as e:
            print(f"Error parsing argument structure: {e}")
            return []
            
        return arguments

    async def process_batch(self, comments: List[str], topic: str) -> List[Argument]:
        # First pass: Get initial argument structure
        initial_prompt = await self._create_initial_prompt(comments, topic)
        messages = [Message(Role("user"), initial_prompt)]
        response = await self.llm_provider.generate(messages, temperature=0.3)
        initial_arguments = await self._parse_argument_structure(response.content)
        
        # Second pass: Refine and reorganize arguments
        redundancy_prompt = await self._create_redundancy_prompt(initial_arguments)
        messages = [Message(Role("user"), redundancy_prompt)]
        response = await self.llm_provider.generate(messages, temperature=0.2)
        final_arguments = await self._parse_argument_structure(response.content)
        
        return final_arguments

    async def consolidate_arguments(self, argument_lists: List[List[Argument]]) -> List[Argument]:
        # Convert arguments to the format expected by the LLM
        structures = []
        for args in argument_lists:
            structure = "<ARGUMENTS>\n"
            for arg in args:
                structure += f"MAIN: {arg.main_point}\n"
                for sub in arg.subpoints:
                    structure += f"SUB: {sub}\n"
            structure += "</ARGUMENTS>"
            structures.append(structure)
        
        prompt = await self._create_consolidation_prompt(structures)
        messages = [Message(Role("user"), prompt)]
        
        response = await self.llm_provider.generate(messages, temperature=0.3)
        return await self._parse_argument_structure(response.content)

    async def process_all_comments(self, comments: List[str], topic: str) -> List[Argument]:
        # Process comments in batches
        batches = [comments[i:i + self.batch_size] for i in range(0, len(comments), self.batch_size)]
        batch_results = []
        
        for batch in batches:
            result = await self.process_batch(batch, topic)
            batch_results.append(result)
            
        # Consolidate all batch results
        final_arguments = await self.consolidate_arguments(batch_results)
        return final_arguments
    
    def _format_argument(self, argument: Argument) -> str:
        formatted = f"MAIN: {argument.main_point}\n"
        for sub in argument.subpoints:
            formatted += f"SUB: {sub}\n"
        return formatted
    
    async def format_arguments(self, arguments: List[Argument]) -> str:
        formatted = ""
        for arg in arguments:
            formatted += self._format_argument(arg) + "\n"
        return formatted

# Example usage
async def main():
    # config
    api_key = os.getenv("OPENAI_API_KEY")
    model="gpt-4o-2024-11-20"
    openai_provider = OpenAIProvider(api_key, model)
    data_source_path = os.path.join(parent_dir, 'data', 'sources', 'kuntavaalit2021.csv')
    output_path = os.path.join(parent_dir, 'condensation', 'results','results_with_indices', 'version1.txt')
    n_comments = 100
    batch_size = 100
    topic = ""

    # get comments
    df = pd.read_csv(data_source_path)
    comment_indices = df['q9.explanation_fi'].dropna()[:n_comments].index.tolist()
    comments = df.loc[comment_indices, 'q9.explanation_fi'].tolist()
    likert_answers = df.loc[comment_indices, 'q9.answer_fi'].tolist()

    # process arguments
    processor = ArgumentProcessor(openai_provider, batch_size)
    arguments = await processor.process_batch(comments,topic)
    formatted_args = await processor.format_arguments(arguments)

    # print results
    print(formatted_args)

# run the main function
import asyncio
asyncio.run(main())


