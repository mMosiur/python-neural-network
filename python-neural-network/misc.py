def log(msg, lvl=0):
    while(lvl>0):
        msg = "  " + msg
        lvl-=1
    msg = "> " + msg
    print(msg)
    return

def bytes_to_int(bytes):
    result = 0
    for b in bytes: result = result*256 + int(b)
    return result

def num_to_str(c):
    if(c<10):
        return str(c)+"  "
    elif(c<100):
        return str(c)+" "
    else:
        return str(c)
