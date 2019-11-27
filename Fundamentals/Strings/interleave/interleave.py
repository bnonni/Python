def interleave(a, b):
 c = list(a)
 d = list(b)
 j = ''
 if len(a) < len(b):
  n = len(a)
  for i in range(n):
   j += a[i]
   j += b[i]
  j += b[i:]
 else:
  n = len(b)
  for i in range(n):
   j += a[i]
   j += b[i]
  j += a[i:]
 
 print(j)
 
def tests():
 t1 = interleave("HuskyHacker", "bryan")
 # assert (t1 == "HbarcykaenrRank")
 t2 = interleave("bryan", "HuskyHacker")
 # assert (t2 == "bHraycaknerRank")
 
tests()