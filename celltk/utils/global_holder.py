

def counter():
    i = 0
    while True:
        i += 1
        yield i


class Holder(object):
    c = counter()

    def count(self):
        return self.c.__next__()

# global holder
holder = Holder()
