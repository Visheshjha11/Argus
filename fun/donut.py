
import math
import os
import time

A = 0
B = 0

while True:
    os.system("cls" if os.name == "nt" else "clear")

    z = [0] * 1760
    b = [" "] * 1760

    for j in range(0, 628, 7):
        j /= 100

        for i in range(0, 628, 2):
            i /= 100

            c = math.sin(i)
            d = math.cos(j)
            e = math.sin(A)
            f = math.sin(j)
            g = math.cos(A)
            h = d + 2
            D = 1 / (c * h * e + f * g + 5)

            l = math.cos(i)
            m = math.cos(B)
            n = math.sin(B)
            t = c * h * g - f * e

            x = int(40 + 30 * D * (l * h * m - t * n))
            y = int(12 + 15 * D * (l * h * n + t * m))

            o = x + 80 * y

            N = int(
                8 * (
                    (f * e - c * d * g) * m
                    - c * d * e
                    - f * g
                    - l * d * n
                )
            )

            if 0 <= o < 1760 and D > z[o]:
                z[o] = D
                b[o] = ".,-~:;=!*#$@="[max(0, N)]

    print("\033[H", end="")

    for k in range(1760):
        print(b[k], end="\n" if k % 80 == 79 else "")

    A += 0.04
    B += 0.02

    time.sleep(0.03)