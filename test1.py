import sympy as sp

def extended_euclidean(a, b):
  if a == 0:
    return b, 0, 1
  else:
    gcd, x, y = extended_euclidean(b % a, a)
    return gcd, y - (b // a) * x, x
    
def mod_inverse(a, m):
  gcd, x, y = extended_euclidean(a, m)
  if gcd != 1:
    return None
  else:
    return x % m
  
def find_vertical_asymptotes_and_modular_inverse():
  x = sp.symbols('x')
  equation = (x**2 - 4) * sp.log(x - 1)
  solutions = sp.solve(equation, x)
  print(solutions)
  
  function = (x**3 - x**2 - 4*x + 4) / (sp.log(x-1)*(x**2 - 4))
  limit_left = sp.limit(function, x, 2, dir="-")
  limit_right = sp.limit(function, x, 2, dir="+")
  
  if limit_left.is_infinite or limit_right.is_infinite:
    print("There is a vertical asymptote at x = 2.")
    a = 2
    m = 17
    inverse = mod_inverse(a, m)
    print(f"The modular inverse of {a} modulo {m} is: {inverse}")
  else:
    print("There is no vertical asymptote at x = 2.")

find_vertical_asymptotes_and_modular_inverse()

def solve_differential_equation():

    x = sp.symbols('x')

    rhs = 3 * sp.exp(2 * x)

    y = sp.integrate(rhs, x)

    C = sp.symbols('C')

    y = sp.Add(y, C)

    integral = sp.integrate(y, (x, 0, 1))

    return y, integral

y_solution, integral_result = solve_differential_equation()
print("General solution y(x):", y_solution)
print("Definite integral from 0 to 1:", integral_result)


import itertools

# Define the factors of the generating function
factors = [
    [0, 1, 2, 3, 4, 5, 6],  # Represents powers of x in the first factor: x^0, x^1, ..., x^6
    [0, 2, 4, 6],           # Represents powers of x in the second factor: x^0, x^2, x^4, x^6
    [0, 3, 6],              # Represents powers of x in the third factor: x^0, x^3, x^6
    [0, 4],                 # Represents powers of x in the fourth factor: x^0, x^4
    [0, 5]                  # Represents powers of x in the fifth factor: x^0, x^5
]

# Initialize the coefficient of x^6
coefficient = 0

# Iterate over all possible combinations of terms
for combination in itertools.product(*factors):
    # Sum the powers in the combination
    if sum(combination) == 6:
        # Increment the coefficient
        coefficient += 1

print(coefficient)


import math

# 定义函数 f(x)
def f(x):
    if isinstance(x, int) and x > 0:
        return 2 * x + 3
    elif isinstance(x, int) and x < 0:
        return x**2 - 4
    elif x == 0:
        return 5
    else:
        return math.sin(x)

# 生成整数和非整数实数的 x 值列表
integers = list(range(-10, 11))
real_numbers = [i / 10 for i in range(-100, 101) if i % 10 != 0]
x_values = integers + real_numbers

# 初始化计数变量
count1 = 0
count2 = 0
count3 = 0
count4 = 0

print(x_values)
# 遍历 x_values 验证逆否命题
for x in x_values:
    if f(x) != 2 * x + 3 and (x > 0 and x == int(x)):
        count1 += 1
    if f(x) != x**2 - 4 and (x < 0 and x == int(x)):
        count2 += 1
    if f(x) != 5 and x == 0:
        count3 += 1
    if f(x) != math.sin(x) and (x != int(x)):
        count4 += 1

# 输出结果
print("Count 1:", count1)
print("Count 2:", count2)
print("Count 3:", count3)
print("Count 4:", count4)