
def KeyGenerator(m):
   a = m*x+r
   b = 0
   a_inv = 0
   temp = f'{a}{b}{a_inv}'
   return temp


def a_inverse(a, modulus):
   for i in range(modulus+1):
      if i * mod(a, modulus)  == 1:
         return i
      else:
         return 0











key = KeyGenerator(6449)