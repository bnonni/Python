
def compute_average(numbers):
 total = 0
 length = len(numbers)
 for number in numbers:
  total += number
  
 return total/length


compute_average([1,2,3,4,5])


def compute_median(numbers):
 length = len(numbers) / 2
 numbers.sort()
 if length % 2 == 0:
  return (numbers[length] + numbers[length - 1]) / 2
 else:
  return numbers[round(length)]

 
 
 
 
 