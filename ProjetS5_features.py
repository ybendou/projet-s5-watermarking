import numpy as np








def average_dist_per_line(d) :
#input: pytesseract dict
#output: list of average distance between words for each lines
    words_per_line = words_in_lines()
    return [int(np.mean(interword_distances_line_i(i))) for i in range(len(words_per_line))]







def words_in_lines(d) :
#input: pytesseract dict
#output: list of the number of words per line
    words_per_line = []
    l = -1
    for num in d['word_num'] :
        if num == 1 :
            words_per_line.append(1)
            l += 1
        else :
            words_per_line[l] += 1
        return words_per_line



def margin_distance(d,i,left,words_per_lines) :
#input: pytesseract dict, line number and left = boolean, if it's false we're looking for the right margin
#output: margin size
    a = 0
    if not left :
        a = 1
    n = words_per_line[i]
    pos = np.zeros(n)
    start = int(np.sum(words_per_line[:i])) + a*next

    return d['left'][start] #TODO Write something for the right margin



def interword_distances_line_i(d,i,words_per_line) :
#input: pytesseract dict,the line number and a list of the number of words per line
#output: list of distance bewteen each words
    if (i >=len(words_per_line)) :
        print("This line does not exist")
        return -1

    if words_per_line[i] <2 :
        print("Not enough words")
        return -1
    n = words_per_line[i]
    dist = np.zeros(n-1)
    start = int(np.sum(words_per_line[:i]))
    for i in range(n-1) :
        index = start+i
        (x, w) = (d['left'][index], d['width'][index])
        x2 = d['left'][index+1]
        dist[i] = x2 - x - w
    return dist



def line_position(d,i,words_per_line) :
#input: pytesseract dict,the line number and a list of the number of words per line
#output: position of a line computed manually
    n = words_per_line[i]
    pos = np.zeros(n)
    start = int(np.sum(words_per_line[:i]))
    for i in range(n) :
        index = start+i
        (y, h) = (d['top'][index], d['height'][index])
        pos[i] = y
        #pos[i] = (y+h)/2
    return np.mean(pos)


def line_distance(d,i,j,words_per_lines) :
#input: pytesseract dict,the two line numbers and a list of the number of words per line
#output: distance between the two lines
    return abs(line_position(d,i,words_per_line) - line_position(d,j,words_per_line))























