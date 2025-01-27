from typing import List, Dict
import re

class OutputParser:
    """Parses LLM outputs into structured data."""
    
    def parse_arguments(self, text: str) -> List[str]:
        """Parse argument texts from LLM response."""
        arguments = []
        lines = text.split('\n')
        current_arg = []
        
        in_arguments = False
        for line in lines:
            line = line.strip()
            if '<ARGUMENTS>' in line:
                in_arguments = True
                continue
            elif '</ARGUMENTS>' in line:
                break
            elif in_arguments and line.startswith('ARGUMENTTI'):
                if current_arg:
                    arguments.append(' '.join(current_arg))
                    current_arg = []
                current_arg = [line.split(':', 1)[1].strip()]
            elif in_arguments and not line.startswith('Lähteet:') and line:
                current_arg.append(line)
                
        if current_arg:
            arguments.append(' '.join(current_arg))
            
        return arguments

    def parse_source_indices(self, text: str) -> List[List[int]]:
        """Parse source indices for each argument from LLM response."""
        source_indices_per_arg = []
        lines = text.split('\n')
        
        in_arguments = False
        for i, line in enumerate(lines):
            line = line.strip()
            if '<ARGUMENTS>' in line:
                in_arguments = True
                continue
            elif '</ARGUMENTS>' in line:
                break
            elif in_arguments and line.startswith('Lähteet:'):
                # Extract numbers from the line
                numbers_str = line.split(':', 1)[1].strip('[] ')
                if numbers_str:
                    # Extract just the numbers from strings like "Kommentti 8"
                    numbers = []
                    for num_str in numbers_str.split(','):
                        # Find any number in the string
                        matches = re.findall(r'\d+', num_str.strip())
                        if matches:
                            numbers.extend(int(match) for match in matches)
                    source_indices_per_arg.append(numbers)
                else:
                    source_indices_per_arg.append([])
                    
        return source_indices_per_arg
    