def checksurname(sur, end):
    return sur[len(sur) - len(end):] == end

def checkcolor(des, color):
    return color.lower() in des.lower()

print(checksurname('kello', 'lol'))
print(checkcolor('RED', 'red'))