# Есть 2 числа, записанных в виде строки. Задача их сложить не кастуя из в int. Вернуть ответ в виде строки
# Есть набор с кординатами x, y. Задача - есть ли ось симметри

def sum_int_str(val1, val2):
    print(int(val1) + int(val2))
    result = []
    delta = 0
    for l1, l2 in zip(val1[::-1], val2[::-1]):
        tmp_sum = int(l1) + int(l2)
        digit = (tmp_sum + delta) % 10
        delta = digit // 10
        result.append(digit)

    val = val1 if len(val1) > len(val2) else val2

    st_bit = min(len(val1), len(val2))  #?
    for l in val[::-1][st_bit:]:
        tmp_sum = int(l)
        print(tmp_sum)
        digit = (tmp_sum + delta) % 10
        delta = tmp_sum // 10
        result.append(digit)

    if delta:
        result.append(delta)

    return ''.join(map(str, result))[::-1]

# print(sum_int_str('99', '9'))
print([1,2,3,4,5,6,7,8,9][1:5][::-1])