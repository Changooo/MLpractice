class parent:
    
    a = 10
    def __init__(self):
        print("parent")

    def hi(self):
        print("wow")

class Child(parent):
    def __init__(self):
        # super().__init__()
        print(self.a)
    

# k = parent()
# k.hi()
c = Child()
