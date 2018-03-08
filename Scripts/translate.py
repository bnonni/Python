in_code = 'abcdefghijklmnopqrstuvwxyz'
out_code = '9128645!@#$%^&*()/.,;:~|[]'

code = str.maketrans(in_code, out_code)

print('this is encrypted!'.translate(code))
