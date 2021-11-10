
class InputParameters():
    def __init__(self):
        self.sample = "gecko_123438"
        self.z_range = [600, 1000]
        self.coords_2d = tuple([slice(300, 750), slice(300, 850)])
        self.coords_3d = tuple([slice(0, self.z_range[1]-self.z_range[0]), *self.coords_2d])


       
    
    