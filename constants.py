ENGLISH_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
VALID_PUNCS = '?.,;:!\'"(){}[]-'
NUMBERS = '0123456789'
SPECIAL = ' '

NORMLIZER_MAPPER = {
    'é': 'e',
    'á': 'a',
    'í': 'i',
    'ó': 'o',
    'ú': 'u',
    'É': 'E',
    'Á': 'A',
    'Í': 'I',
    'Ó': 'O',
    'Ú': 'U'
}
VALID_CHARS = ENGLISH_CHARS + SPECIAL + VALID_PUNCS + NUMBERS
KEYBOARD_KEYS = [
    'qwertyuiop',
    'asdfghjkl',
    'zxcvbnm'
]
KEYBOARD_BLANK = ' '
