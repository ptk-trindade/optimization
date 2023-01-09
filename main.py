import numpy as np


def dom(x, y):
    if x*y*(1-x)*(1-y) <= 0:
        return False
    return True


def f(x, y):
    return np.arctan(-np.log(x*(1-x)*y*(1-y)))


def grad_f(x, y):
    var = np.log(x*y*(1-x)*(1-y))+1

    # grad x
    g1 = (2*x - 1)/(x*(var*var)*(1-x))

    # grad y
    g2 = (2*y - 1)/(y*(var*var)*(1-y))

    # print(f"grad({x}, {y}):\n{g1} {g2}")
    return np.array([g1, g2])


def hessian_f(x, y):
    x2 = x*x
    y2 = y*y

    ln = np.log(x*y*(1-x)*(1-y))
    ln2 = ln*ln

    2*ln*(2*x-1)*(2*y-1)/(x*y*(1-x)*(1-y)*(ln2 + 1)**2)

    g11 = (-2*x2 * ln2 - 2*x2 - 8*x2 * ln + 2*x * ln2 + 2*x + 8*x*ln - ln2 - 2*ln - 1) / (x2 * (ln2 + 1)**2 * (1-x)**2)
    g12 = 2*ln*(2*x-1)*(2*y-1)/(x*y*(1-x)*(1-y)*(ln2 + 1)**2) #(2*ln*(1 - 2*x)*(1 - 2*y)) / (x*y*(1-x)*(1-y)*(ln2 + 1)**2) # g21 = g12
    g22 = (-2*y2 * ln2 - 2*y2 - 8*y2 * ln + 2*y * ln2 + 2*y + 8*y*ln - ln2 - 2*ln - 1) / (y2 * (ln2 + 1)**2 * (1-y)**2)

    # print(f"hessian({x}, {y}):\n{g11} {g12}\n{g12} {g22}")
    return np.array([[g11, g12], [g12, g22]])


# Armijo's method
# x, y: current point
# d: direction
# grad: gradient at current point
# max_step: maximum step size
# beta: step size reduction factor
# sigma: gradient "relief" factor
def armijo(x, y, d, grad, max_step = 100, beta = 0.8, sigma = 0.1):
    f_xy = f(x, y)
    # grad = grad_f(x, y)
    # d = -(grad/np.linalg.norm(grad))

    # print(f"grad:\n{grad} \n\n d:\n{d}")
    upper_limit = sigma*np.dot(d, grad)

    # print("upper limit (should be negative): ", upper_limit)
    step = max_step
    k = 1
    min_step = 1/(10**16)
    # print("f_xy:", f_xy, x, y)
    while step > min_step and f(x + step*d[0], y + step*d[1]) > f_xy - step*upper_limit:
        # print("f'_xy:", f(x + step*d[0], y + step*d[1]), x + step*d[0], y + step*d[1])
        # input()
        step = beta*step
        k += 1

    if step <= min_step:
        print("ERROR: Armijo did not converge")
        print(f"iterations = {k}, beta = {beta}, sigma = {sigma}, step = {step}")
        input("Press enter to continue...")

    return step, k



# Gradient descent
# x, y: initial point
# iter_limit: maximum number of iterations
def gradient_descent(x, y, iter_limit):
    k = 0
    grad = grad_f(x, y)
    armijo_k = 0
    # print("x\ty")
    while k < iter_limit and np.linalg.norm(grad) > 0.000001: # gradient magnitude > 0.0001
        d = -(grad/np.linalg.norm(grad))    # d = -grad/|grad|
        
        step, arm_k = armijo(x, y, d, grad)

        armijo_k += arm_k

        x = x + step*d[0]
        y = y + step*d[1]
        # print(x, y)

        grad = grad_f(x, y)
        k += 1

    if k == iter_limit:
        print("Gradient descent did not converge")


    print(f"k: {k}, armijo_k: {armijo_k}")
    return x, y

# Newton's method
# x, y: initial point
# iter_limit: maximum number of iterations
def newton_method(x, y, iter_limit):
    k = 0
    grad = grad_f(x, y)
    hes = hessian_f(x, y)
    armijo_k = 0
    # print("x\ty")
    while k < iter_limit and np.linalg.norm(grad) > 0.00001: # gradient magnitude > 0.0001
        d = np.linalg.solve(hes, grad) # d = H^-1 * grad
        d = d/np.linalg.norm(d) # normalize d
        # print("d:", d)

        # print("x:", x, "y:", y, "d:", d, "grad:", grad)
        step, arm_k = armijo(x, y, d, grad)
        # print("arm_k:", arm_k)
        armijo_k += arm_k
        # print("step:", step)
        
        x = x + step*d[0]
        y = y + step*d[1]
        # print(x, y)
        
        grad = grad_f(x, y)
        hes = hessian_f(x, y)
        k += 1
    
    if k == iter_limit:
        print("Newton's method did not converge")

    print(f"k: {k}, armijo_k: {armijo_k}")
    return x, y


# DFP method
# x, y: initial point
# iter_limit: maximum number of iterations
def dfp_method(x, y, iter_limit):
    # return x, y
    k = 0
    grad = grad_old = grad_f(x, y)
    hes = np.identity(2)
    armijo_k = 0
    # print("x\ty")
    while k < iter_limit and np.linalg.norm(grad) > 0.00001: # gradient magnitude > 0.0001

        d = np.dot(-hes, grad)
        d = d/np.linalg.norm(d) # normalize d
        step, arm_k = armijo(x, y, d, grad)
        armijo_k += arm_k

        x_old = x
        y_old = y

        x = x + step*d[0]
        y = y + step*d[1]

        grad_old = grad
        grad = grad_f(x, y)

        p = np.array([x - x_old, y - y_old])
        q = grad - grad_old

        hes = hes + np.outer(p, p)/np.dot(p, q) - np.matmul(np.matmul(hes, np.outer(q, q)), hes)/np.dot(q, np.dot(hes, q))


    if k == iter_limit:
        print("Newton's method did not converge")

    print(f"k: {k}, armijo_k: {armijo_k}")
    return x, y


def main():
    while True:
        print("Escolha um ponto inicial para x e y")
        x_str = input("x: ")
        y_str = input("y: ")

        try:
            x = float(x_str)
            y = float(y_str)
        except:
            print("Não foi possível converter os valores para float")
            continue

        if not dom(x, y):
            print("Coorderadas fora do domínio")
            continue


        print("\nInsira o id do método a ser utilizado:")
        print("1 - Método do Gradiente")
        print("2 - Método de Newton")
        print("3 - Quase Newton")
        print("0 - Sair")

        method = input("Método: ")

        if method == "1":
            print("Método do Gradiente:")
            print(f"Ponto inicial:\tf({x}, {y}) = {f(x, y)}")
            x, y = gradient_descent(x, y, 10**6)
            print(f"Minimo encontrado:\tf({x}, {y}) = {f(x, y)}")

        elif method == "2":
            print("Newton's method:")
            print(f"Ponto inicial:\tf({x}, {y}) = {f(x, y)}")
            x, y = newton_method(x, y, 10**6)
            print(f"Minimo encontrado:\tf({x}, {y}) = {f(x, y)}")


        elif method == "3":
            print("Quase Newton:")
            print(f"Ponto inicial:\tf({x}, {y}) = {f(x, y)}")
            x, y = dfp_method(x, y, 10**6)
            print(f"Minimo encontrado:\tf({x}, {y}) = {f(x, y)}")

        elif method == "0":
            print("Saindo...")
            return

        else:
            print("Invalid input")


main()
