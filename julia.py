print("hello?")

from mpi4py.futures import MPIPoolExecutor

x0, x1, w = -2.0, +2.0, 640*2
y0, y1, h = -1.5, +1.5, 480*2
dx = (x1 - x0) / w
dy = (y1 - y0) / h

c = complex(0, 0.65)

def julia(x, y):
    z = complex(x, y)
    n = 255
    while abs(z) < 3 and n > 1:
        z = z**2 + c
        n -= 1
    return n

def julia_line(k):

    print("I got:" + str(k))
    line = bytearray(w)
    y = y1 - k * dy
    for j in range(w):
        x = x0 + j * dx
        line[j] = julia(x, y)
    return line


def say_hi(k):

    print("Hello, world! You gave me" + str(k))


    return(k**2)

if __name__ == '__main__':

    print("Starting main.")


    with MPIPoolExecutor() as executor:


        print("About to map")

        output = executor.map(say_hi, range(4))

        for thing in output:
            print("The output is: ", thing)


        # image = executor.map(julia_line, range(h))
        # with open('julia.pgm', 'wb') as f:
        #     f.write(b'P5 %d %d %d\n' % (w, h, 255))
        #     for line in image:
        #         f.write(line)
