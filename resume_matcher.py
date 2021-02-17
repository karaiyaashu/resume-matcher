from sklearn.metrics.pairwise import cosine_similarity
from pdfTotext import convert_pdf_to_txt
from sklearn.feature_extraction.text import CountVectorizer

resume = convert_pdf_to_txt('test.pdf')
with open('jd.txt') as f:
    jd = f.read()
# print(jd)

text = [resume, jd]


cv = CountVectorizer()
count_matrix = cv.fit_transform(text)


# print(cosine_similarity(count_matrix))

matchPercentage = cosine_similarity(count_matrix)[0][1]*100

matchPercentage = round(matchPercentage, 2)
print(f'you resume matches {matchPercentage} of JD')
