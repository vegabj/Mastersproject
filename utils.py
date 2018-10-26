# Cache for functions
def cache(func):
  saved = {}
  @wraps(func)
  def newfunc(*args):
    if args in saved:
      return newfunc(*args)
    result = func(*args)
    saved[args] = result
    return result
  return newfunc
