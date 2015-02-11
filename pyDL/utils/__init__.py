

def _flaten(x, rval):
    if isinstance(x, (tuple, list)):
        for xitem in x:
            _flaten(xitem, rval)
    else:
        rval.append(x)

def flatten(x):
    rval = []
    _flaten(x, rval)
    return tuple(rval)
    

def check_type_constraints(objects, types):
    rval = True
    if isinstance(objects, (tuple, list)):
        for obj in objects:
            rval = rval and isinstance(obj, types)
    else:
        rval = isinstance(objects, types)
    
    return rval
    
