import sys, json, os

def create_section(number: int) -> str:
    return f'\t\\section*{{Zadanie {number}}}\n'

def create_item(lab_nr: int, first_line: int, last_line: int) -> str:
    return f'\t\t\\itemcl[]{{}}{{\nlab{lab_nr}.py}}{{{first_line}}}{{{last_line}}}\n'

def create_image(image_path: str, size: float = 0.6) -> str:
    return f'\t\t\t\\outputImg{{{size}\\textwidth}}{{{image_path}}}'

def create_text(text_path: str) -> str:
    return f'\t\t\t\\outputTxt{{{text_path}}}'

argv = [__file__, 6]



if __name__ == '__main__':
    document = open('preambule/pso_template.tex').read()
    number = argv[1]
    code = open(f'lab{argv[1]}.py').read().split('\n')

    images = os.listdir(f'images/lab{number}')
    snippets = os.listdir(f'snippets/lab{number}')

    section = 0
    subsection = 0
    first_line = 0
    last_line = 0
    text = ''

    for i, line in enumerate(code):

        if line[0] == '#':
            first_line = i + 1
            sections = line.split(' ')[1].split('.')[:2]
            if int(sections[0]) > section:
                section  


    
