import re
import subprocess

# Token types
TOKEN_TYPES = [
    ('NUMBER', r'\d+(\.\d+)?'),
    ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('OPERATOR', r'[+\-*/]'),
    ('RELATIONAL_OPERATOR', r'at|o|==|≠|>|<|≤|≥'),
    ('PAREN_OPEN', r'\('),
    ('PAREN_CLOSE', r'\)'),
    ('ASSIGN', r'='),
    ('COLON', r':'),
    ('COMMA', r','),
    ('STRING', r'"(?:[^"\\]|\\.|\\n)*"|“(?:[^”\\]|\\.|\\n)*”'),
    ('WHITESPACE', r'\s+'),
]

# Keywords
KEYWORDS = {
    'kung': 'IF',
    'sino': 'ELSE',
    'kundi': 'ELSE',
    'kundi sino': 'ELIF',
    'habang': 'WHILE',
    'para': 'FOR',
    'basahin': 'INPUT',
    'ilabas': 'PRINT',
    'int': 'INT',
    'float': 'FLOAT',
    'str': 'STR',
    'at': 'AND',
    'o': 'OR',
    '==': 'EQUAL',
    '≠': 'NOT_EQUAL',
    '>': 'GREATER',
    '<': 'LESS',
    '≤': 'LESS_EQUAL',
    '≥': 'GREATER_EQUAL',
}

class Node:
    def __init__(self, type, value, children=None):
        self.type = type
        self.value = value
        self.children = children or []

def tokenize(code):
    tokens = []
    while code:
        for token_type, pattern in TOKEN_TYPES:
            match = re.match(pattern, code)
            if match:
                value = match.group(0)
                if token_type == 'IDENTIFIER' and value in KEYWORDS:
                    tokens.append((KEYWORDS[value], value))
                elif token_type == 'STRING':
                    value = value.replace('“', '"').replace('”', '"')
                    tokens.append((token_type, value))
                elif token_type != 'WHITESPACE':
                    tokens.append((token_type, value))
                code = code[len(value):]
                print(f"Tokenized: {token_type} - {value}")
                break
        else:
            raise SyntaxError(f"Unexpected character: {code[0]}")
    return tokens

def parse(tokens):
    def parse_expression():
        if tokens[0][0] == 'NUMBER':
            value = float(tokens[0][1])
            tokens.pop(0)
            return Node('NUMBER', value)
        elif tokens[0][0] == 'IDENTIFIER':
            name = tokens[0][1]
            tokens.pop(0)
            if tokens[0][0] == 'PAREN_OPEN':
                tokens.pop(0)
                args = []
                while tokens[0][0] != 'PAREN_CLOSE':
                    args.append(parse_expression())
                    if tokens[0][0] == 'COMMA':
                        tokens.pop(0)
                tokens.pop(0)  # Pop PAREN_CLOSE
                return Node('CALL', name, args)
            return Node('IDENTIFIER', name)
        elif tokens[0][0] == 'PAREN_OPEN':
            tokens.pop(0)
            expr = parse_expression()
            tokens.pop(0)  # Pop PAREN_CLOSE
            return expr
        elif tokens[0][0] == 'STRING':
            value = tokens[0][1]
            tokens.pop(0)
            return Node('STRING', value)
        elif tokens[0][0] == 'RELATIONAL_OPERATOR':
            operator = tokens[0][1]
            tokens.pop(0)
            left = parse_expression()
            right = parse_expression()
            return Node('RELATIONAL_OPERATOR', operator, [left, right])
        raise SyntaxError("Unexpected token: " + tokens[0][1])

    def parse_term():
        if tokens[0][0] in ['IDENTIFIER', 'NUMBER', 'STRING']:
            return parse_expression()
        elif tokens[0][0] == 'PAREN_OPEN':
            tokens.pop(0)
            expr = parse_expression()
            tokens.pop(0)  # Pop PAREN_CLOSE
            return expr
        raise SyntaxError("Unexpected token: " + tokens[0][1])

    def parse_expression_list():
        expressions = []
        expressions.append(parse_expression())
        while tokens and tokens[0][0] == 'COMMA':
            tokens.pop(0)
            expressions.append(parse_expression())
        return expressions

    def parse_statement():
        if tokens[0][0] == 'IF':
            tokens.pop(0)  # Pop 'IF'
            print("Parsing IF statement")
            condition = parse_expression()  # Parse the condition
            print("Parsed condition:", condition.value)
            if tokens[0][0] != 'COLON':
                raise SyntaxError(f"Expected ':' after condition, got: {tokens[0][1]}")
            tokens.pop(0)  # Pop COLON
            print("Parsed COLON")
            body = []
            while tokens and tokens[0][0] not in ('ELSE', 'ELIF', 'NEWLINE'):
                body.append(parse_statement())  # Parse the body statements
                if tokens and tokens[0][0] == 'NEWLINE':
                    tokens.pop(0)  # Consume the newline if present
            else_if_list = []
            while tokens and tokens[0][0] == 'ELIF':
                else_if_list.append(parse_else_if())  # Handle ELIFs
            else_part = None
            if tokens and tokens[0][0] == 'ELSE':
                tokens.pop(0)  # Pop 'ELSE'
                else_part = parse_block()  # Parse the ELSE body
            return Node('IF', None, [condition, body, else_if_list, else_part])
        elif tokens[0][0] == 'PRINT':
            tokens.pop(0)
            args = parse_expression_list()
            return Node('PRINT', None, args)
        elif tokens[0][0] == 'WHILE':
            tokens.pop(0)
            print("Parsing WHILE statement")
            condition = parse_expression()
            print("Parsed condition:", condition.value)
            if tokens[0][0] != 'COLON':
                raise SyntaxError(f"Expected ':' after condition, got: {tokens[0][1]}")
            tokens.pop(0)  # Pop COLON
            print("Parsed COLON")
            body = parse_block()
            return Node('WHILE', None, [condition, body])
        elif tokens[0][0] == 'IDENTIFIER':
            name = tokens[0][1]
            tokens.pop(0)
            if tokens[0][0] == 'ASSIGN':
                tokens.pop(0)
                value = parse_expression()
                return Node('ASSIGN', name, [value])
            raise SyntaxError("Unexpected token: " + tokens[0][1])
        elif tokens[0][0] == 'INPUT':
            tokens.pop(0)
            prompt = parse_expression()
            return Node('INPUT', None, [prompt])
        raise SyntaxError("Unexpected token: " + tokens[0][1])

    def parse_block():
        statements = []
        while tokens and tokens[0][0] != 'NEWLINE':
            statements.append(parse_statement())
        return Node('BLOCK', None, statements)

    def parse_else_if_list():
        else_if_list = []
        while tokens and tokens[0][0] == 'ELIF':
            tokens.pop(0)
            print("Parsing ELIF statement")
            condition = parse_expression()
            print("Parsed condition:", condition.value)
            if tokens[0][0] != 'COLON':
                raise SyntaxError(f"Expected ':' after condition, got: {tokens[0][1]}")
            tokens.pop(0)  # Pop COLON
            print("Parsed COLON")
            body = parse_block()
            else_if_list.append(Node('ELIF', None, [condition, body]))
        return else_if_list

    print("Parsing tokens")
    ast = parse_block()  # Start parsing from the block of statements
    return ast

def generate_c_code(ast):
    def generate_expression(node):
        if node.type == 'NUMBER':
            return str(node.value)
        elif node.type == 'IDENTIFIER':
            return node.value
        elif node.type == 'STRING':
            return f'"{node.value[1:-1]}"'
        elif node.type == 'CALL':
            args = ', '.join(generate_expression(arg) for arg in node.children)
            if node.value == 'int':
                return f'(int) {args}'
            elif node.value == 'float':
                return f'(float) {args}'
            elif node.value == 'basahin':
                return 'atof(gets(NULL))'
            elif node.value == 'ilabas':
                args = ', '.join(f'"{arg.value[1:-1]}"' if arg.type == 'STRING' else generate_expression(arg) for arg in node.children)
                return f'printf({args});'
            else:
                raise ValueError(f"Unknown function: {node.value}")
        elif node.type == 'OPERATOR':
            return f'({generate_expression(node.children[0])} {node.value} {generate_expression(node.children[1])})'
        elif node.type == 'RELATIONAL_OPERATOR':
            return f'({generate_expression(node.children[0])} {node.value} {generate_expression(node.children[1])})'
        raise ValueError(f"Unknown node type: {node.type}")

    def generate_statement(node):
        if node.type == 'ASSIGN':
            return f'{node.value} = {generate_expression(node.children[0])};'
        elif node.type == 'PRINT':
            args = ', '.join(f'"{arg.value[1:-1]}"' if arg.type == 'STRING' else generate_expression(arg) for arg in node.children)
            return f'printf({args});'
        elif node.type == 'INPUT':
            return f'{node.children[0].value} = atof(gets(NULL));'
        elif node.type == 'IF':
            condition = generate_expression(node.children[0])
            body = '\n'.join(generate_statement(stmt) for stmt in node.children[1].children)
            else_if_list = '\n'.join(generate_statement(stmt) for stmt in node.children[2])
            else_part = '\n'.join(generate_statement(stmt) for stmt in node.children[3].children) if node.children[3] else ''
            return f'if ({condition}) {{\n{body}\n}}\n{else_if_list}\n{else_part}'
        elif node.type == 'WHILE':
            condition = generate_expression(node.children[0])
            body = '\n'.join(generate_statement(stmt) for stmt in node.children[1].children)
            return f'while ({condition}) {{\n{body}\n}}'
        elif node.type == 'BLOCK':
            return '\n'.join(generate_statement(stmt) for stmt in node.children)
        raise ValueError(f"Unknown node type: {node.type}")

    c_code = generate_statement(ast)
    return f'#include <stdio.h>\n#include <stdlib.h>\n\nint main() {{\n{c_code}\nreturn 0;\n}}'

def compile_sari(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        sari_code = f.read()
    
    print("Compiling Sari code")
    tokens = tokenize(sari_code)  # Tokenize the Sari code
    print("Tokens:", tokens)
    ast = parse(tokens)  # Pass the tokens to parse
    print("AST:", ast)
    c_code = generate_c_code(ast)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(c_code)
    
    # Compile the generated C code
    subprocess.run(['gcc', '-o', 'sari_program', output_file], check=True)

if __name__ == "__main__":
    compile_sari('main.sari', 'main.c')
    print("Compilation successful. Run the program using: sari_program")