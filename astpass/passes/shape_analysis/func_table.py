def uop_generic(a):
    return a

def binop_generic(left, right):
    if left == right:
        return left
    elif left == ():
        return right
    elif right == ():
        return left
    else:
        raise NotImplementedError("Shape broadcasting is not implemented")
    
def compare_generic(left, right):
    return binop_generic(left, right)

def ifexp_generic(test, body, orelse):
    assert test == () and body == orelse
    return body
    
def matmul_generic(left, right):
    if not (len(left) > 0 and len(right) > 0):
        raise RuntimeError("Matmul cannot happen on scalar operands")

    if left[-1] != right[0]:
        raise RuntimeError(f"Mismatched contracting dimension found for matmul: {left[-1]} and {right[0]}")
    return left[:-1] + right[1:]

def range(*args):
    '''
    The shape of range cannot be determined by the shape of its arguments.
    So simply return a None here, need another pass to pass the values of 
    the arguments.
    '''
    return None 

def slice(low, up, step):
    # ---- Validate inputs ----
    for name, arg in [("low", low), ("up", up), ("step", step)]:
        if not isinstance(arg, (int, type(None), str)):
            raise TypeError(f"{name} must be int, str, or None")

    step = 1 if step is None else step
    if step != 1:
        raise RuntimeError("Non-1 step is not supported in slice")

    low = 0 if low is None else low

    if isinstance(low, int) and low < 0:
        raise ValueError("Slice lower bound must be non-negative")

    # ---- Dispatch rules ----
    table = {
        (int, int): lambda low, up: (up - low,),
        (int, type(None)): lambda low, up: (None,) if low == 0 else (-low,),
        (int, str): lambda low, up: (f":{up}",) if low == 0 else (f"{low}:{up}",),
        (str, int): lambda low, up: (f"{low}:{up}",),
        (str, type(None)): lambda low, up: (f"{low}:",),
        (str, str): lambda low, up: (f"{low}:{up}",),
    }

    key = (type(low), type(up))
    if key not in table:
        raise NotImplementedError(f"Unsupported slice shape: low={low}, up={up}")

    return table[key](low, up)

def subscript(base, indices):
    shape = []
    for i, idx in enumerate(indices):
        # Case 1: scalar index
        if idx == ():
            pass
        elif len(idx) == 1:
            size = idx[0]
            if not isinstance(size, (int, type(None), str)):
                raise TypeError("A shape dimension must be int, str, or None")
            
            table = {
                int: lambda size: size if size >= 0 else base[i] + size,
                type(None): lambda size: base[i],
                str: lambda size: size
            }

            key = type(size)
            shape.append(table[key](size))
        else:
            assert False, "Should not reach here"
    shape += base[len(indices):]
    return tuple(shape)

def numpy_sin(a):
    return uop_generic(a)

def numpy_cos(a):
    return uop_generic(a)

def numpy_tan(a):
    return uop_generic(a)

def numpy_sinh(a):
    return uop_generic(a)

def numpy_round(a, decimals=()):
    return uop_generic(a)

def numpy_rint(a):
    return uop_generic(a)

def numpy_log(a):
    return uop_generic(a)

def numpy_exp(a):
    return uop_generic(a)

def numpy_sqrt(a):
    return uop_generic(a)

def numpy_pow(a, b):
    assert len(b) == 0 or len(b) == 1 or len(b) == len(a)
    return a

def numpy_power(a, b):
    return numpy_pow(a, b)

def numpy_add(a, b):
    return binop_generic(a, b)

def numpy_subtract(a, b):
    return binop_generic(a, b)

def numpy_multiply(a, b):
    return binop_generic(a, b)

def numpy_divide(a, b):
    return binop_generic(a, b)

def numpy_minimum(a, b):
    return binop_generic(a, b)

def numpy_maximum(a, b):
    return binop_generic(a, b)

def numpy_reduce_generic(a, axis=None):
    if axis != None:
        assert isinstance(axis, int)
        if not (axis >= 0 and axis < len(a)):
            raise ValueError(f"axis={axis} must be in range [0, {len(a)})")
        
        a = list(a)
        del a[axis]
        return tuple(a)
    else:
        return ()

## Built-in functions
def pow(a, b):
    return numpy_pow(a, b)

def min(a, b):
    assert a == () and b == ()
    return ()

def max(a, b):
    assert a == () and b == ()
    return ()

def erf(a):
    return uop_generic(a)