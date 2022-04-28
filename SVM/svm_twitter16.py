import os
import json
import csv
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from utils import preprocess_en_text


np.random.seed(1111)

# df = pd.read_csv("pheme_english.csv", encoding="utf-8")
# indo_stopwords = ["ada","adalah","adanya","adapun","agak","agaknya","agar","akan","akankah","akhir","akhiri","akhirnya","aku","akulah","amat","amatlah","anda","andalah","antar","antara","antaranya","apa","apaan","apabila","apakah","apalagi","apatah","artinya","asal","asalkan","atas","atau","ataukah","ataupun","awal","awalnya","bagai","bagaikan","bagaimana","bagaimanakah","bagaimanapun","bagi","bagian","bahkan","bahwa","bahwasanya","baik","bakal","bakalan","balik","banyak","bapak","baru","bawah","beberapa","begini","beginian","beginikah","beginilah","begitu","begitukah","begitulah","begitupun","bekerja","belakang","belakangan","belum","belumlah","benar","benarkah","benarlah","berada","berakhir","berakhirlah","berakhirnya","berapa","berapakah","berapalah","berapapun","berarti","berawal","berbagai","berdatangan","beri","berikan","berikut","berikutnya","berjumlah","berkali-kali","berkata","berkehendak","berkeinginan","berkenaan","berlainan","berlalu","berlangsung","berlebihan","bermacam","bermacam-macam","bermaksud","bermula","bersama","bersama-sama","bersiap","bersiap-siap","bertanya","bertanya-tanya","berturut","berturut-turut","bertutur","berujar","berupa","besar","betul","betulkah","biasa","biasanya","bila","bilakah","bisa","bisakah","boleh","bolehkah","bolehlah","buat","bukan","bukankah","bukanlah","bukannya","bulan","bung","cara","caranya","cukup","cukupkah","cukuplah","cuma","dahulu","dalam","dan","dapat","dari","daripada","datang","dekat","demi","demikian","demikianlah","dengan","depan","di","dia","diakhiri","diakhirinya","dialah","diantara","diantaranya","diberi","diberikan","diberikannya","dibuat","dibuatnya","didapat","didatangkan","digunakan","diibaratkan","diibaratkannya","diingat","diingatkan","diinginkan","dijawab","dijelaskan","dijelaskannya","dikarenakan","dikatakan","dikatakannya","dikerjakan","diketahui","diketahuinya","dikira","dilakukan","dilalui","dilihat","dimaksud","dimaksudkan","dimaksudkannya","dimaksudnya","diminta","dimintai","dimisalkan","dimulai","dimulailah","dimulainya","dimungkinkan","dini","dipastikan","diperbuat","diperbuatnya","dipergunakan","diperkirakan","diperlihatkan","diperlukan","diperlukannya","dipersoalkan","dipertanyakan","dipunyai","diri","dirinya","disampaikan","disebut","disebutkan","disebutkannya","disini","disinilah","ditambahkan","ditandaskan","ditanya","ditanyai","ditanyakan","ditegaskan","ditujukan","ditunjuk","ditunjuki","ditunjukkan","ditunjukkannya","ditunjuknya","dituturkan","dituturkannya","diucapkan","diucapkannya","diungkapkan","dong","dua","dulu","empat","enggak","enggaknya","entah","entahlah","guna","gunakan","hal","hampir","hanya","hanyalah","hari","harus","haruslah","harusnya","hendak","hendaklah","hendaknya","hingga","ia","ialah","ibarat","ibaratkan","ibaratnya","ibu","ikut","ingat","ingat-ingat","ingin","inginkah","inginkan","ini","inikah","inilah","itu","itukah","itulah","jadi","jadilah","jadinya","jangan","jangankan","janganlah","jauh","jawab","jawaban","jawabnya","jelas","jelaskan","jelaslah","jelasnya","jika","jikalau","juga","jumlah","jumlahnya","justru","kala","kalau","kalaulah","kalaupun","kalian","kami","kamilah","kamu","kamulah","kan","kapan","kapankah","kapanpun","karena","karenanya","kasus","kata","katakan","katakanlah","katanya","ke","keadaan","kebetulan","kecil","kedua","keduanya","keinginan","kelamaan","kelihatan","kelihatannya","kelima","keluar","kembali","kemudian","kemungkinan","kemungkinannya","kenapa","kepada","kepadanya","kesampaian","keseluruhan","keseluruhannya","keterlaluan","ketika","khususnya","kini","kinilah","kira","kira-kira","kiranya","kita","kitalah","kok","kurang","lagi","lagian","lah","lain","lainnya","lalu","lama","lamanya","lanjut","lanjutnya","lebih","lewat","lima","luar","macam","maka","makanya","makin","malah","malahan","mampu","mampukah","mana","manakala","manalagi","masa","masalah","masalahnya","masih","masihkah","masing","masing-masing","mau","maupun","melainkan","melakukan","melalui","melihat","melihatnya","memang","memastikan","memberi","memberikan","membuat","memerlukan","memihak","meminta","memintakan","memisalkan","memperbuat","mempergunakan","memperkirakan","memperlihatkan","mempersiapkan","mempersoalkan","mempertanyakan","mempunyai","memulai","memungkinkan","menaiki","menambahkan","menandaskan","menanti","menanti-nanti","menantikan","menanya","menanyai","menanyakan","mendapat","mendapatkan","mendatang","mendatangi","mendatangkan","menegaskan","mengakhiri","mengapa","mengatakan","mengatakannya","mengenai","mengerjakan","mengetahui","menggunakan","menghendaki","mengibaratkan","mengibaratkannya","mengingat","mengingatkan","menginginkan","mengira","mengucapkan","mengucapkannya","mengungkapkan","menjadi","menjawab","menjelaskan","menuju","menunjuk","menunjuki","menunjukkan","menunjuknya","menurut","menuturkan","menyampaikan","menyangkut","menyatakan","menyebutkan","menyeluruh","menyiapkan","merasa","mereka","merekalah","merupakan","meski","meskipun","meyakini","meyakinkan","minta","mirip","misal","misalkan","misalnya","mula","mulai","mulailah","mulanya","mungkin","mungkinkah","nah","naik","namun","nanti","nantinya","nyaris","nyatanya","oleh","olehnya","pada","padahal","padanya","pak","paling","panjang","pantas","para","pasti","pastilah","penting","pentingnya","per","percuma","perlu","perlukah","perlunya","pernah","persoalan","pertama","pertama-tama","pertanyaan","pertanyakan","pihak","pihaknya","pukul","pula","pun","punya","rasa","rasanya","rata","rupanya","saat","saatnya","saja","sajalah","saling","sama","sama-sama","sambil","sampai","sampai-sampai","sampaikan","sana","sangat","sangatlah","satu","saya","sayalah","se","sebab","sebabnya","sebagai","sebagaimana","sebagainya","sebagian","sebaik","sebaik-baiknya","sebaiknya","sebaliknya","sebanyak","sebegini","sebegitu","sebelum","sebelumnya","sebenarnya","seberapa","sebesar","sebetulnya","sebisanya","sebuah","sebut","sebutlah","sebutnya","secara","secukupnya","sedang","sedangkan","sedemikian","sedikit","sedikitnya","seenaknya","segala","segalanya","segera","seharusnya","sehingga","seingat","sejak","sejauh","sejenak","sejumlah","sekadar","sekadarnya","sekali","sekali-kali","sekalian","sekaligus","sekalipun","sekarang","sekecil","seketika","sekiranya","sekitar","sekitarnya","sekurang-kurangnya","sekurangnya","sela","selagi","selain","selaku","selalu","selama","selama-lamanya","selamanya","selanjutnya","seluruh","seluruhnya","semacam","semakin","semampu","semampunya","semasa","semasih","semata","semata-mata","semaunya","sementara","semisal","semisalnya","sempat","semua","semuanya","semula","sendiri","sendirian","sendirinya","seolah","seolah-olah","seorang","sepanjang","sepantasnya","sepantasnyalah","seperlunya","seperti","sepertinya","sepihak","sering","seringnya","serta","serupa","sesaat","sesama","sesampai","sesegera","sesekali","seseorang","sesuatu","sesuatunya","sesudah","sesudahnya","setelah","setempat","setengah","seterusnya","setiap","setiba","setibanya","setidak-tidaknya","setidaknya","setinggi","seusai","sewaktu","siap","siapa","siapakah","siapapun","sini","sinilah","soal","soalnya","suatu","sudah","sudahkah","sudahlah","supaya","tadi","tadinya","tahu","tahun","tak","tambah","tambahnya","tampak","tampaknya","tandas","tandasnya","tanpa","tanya","tanyakan","tanyanya","tapi","tegas","tegasnya","telah","tempat","tengah","tentang","tentu","tentulah","tentunya","tepat","terakhir","terasa","terbanyak","terdahulu","terdapat","terdiri","terhadap","terhadapnya","teringat","teringat-ingat","terjadi","terjadilah","terjadinya","terkira","terlalu","terlebih","terlihat","termasuk","ternyata","tersampaikan","tersebut","tersebutlah","tertentu","tertuju","terus","terutama","tetap","tetapi","tiap","tiba","tiba-tiba","tidak","tidakkah","tidaklah","tiga","tinggi","toh","tunjuk","turut","tutur","tuturnya","ucap","ucapnya","ujar","ujarnya","umum","umumnya","ungkap","ungkapnya","untuk","usah","usai","waduh","wah","wahai","waktu","waktunya","walau","walaupun","wong","yaitu","yakin","yakni","yang"]


# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
#
# stemmer = SnowballStemmer('english')
#
def tokenize(text):
    #print(text)
    tknzr = TweetTokenizer()
    tokens = [word for word in tknzr.tokenize(text)]
    # stems = [stemmer.stem(item) for item in tokens]
    # print(stems)
    return tokens



from sklearn import svm
from sklearn.metrics import classification_report


# train_event_list = ['charliehebdo-all-rnr-threads', 'ebola-essien-all-rnr-threads', 'ferguson-all-rnr-threads', 'germanwings-crash-all-rnr-threads',
#                   'gurlitt-all-rnr-threads', 'ottawashooting-all-rnr-threads', 'prince-toronto-all-rnr-threads', 'putinmissing-all-rnr-threads',
#                   'sydneysiege-all-rnr-threads']
#
# test_event_list = ['charliehebdo-all-rnr-threads',  'ferguson-all-rnr-threads',
#                       'germanwings-crash-all-rnr-threads', 'ottawashooting-all-rnr-threads',
#                       'sydneysiege-all-rnr-threads']

langs = ["en", "id", "vi", "th", "ms"]

for lang in langs:
#change this accordingly
    train_lang = lang
    test_lang = lang
    features = [10000, 20000, 30000]
    C_list = [0.001, 0.1, 1, 100, 1000]
    gamma_list = [0.001, 0.01, 0.1, 1, 10]
    for num_features in features:
        for C_value in C_list:
            for gamma_value in gamma_list:
                results = []
                raw_results = []
                print(num_features)

                for split in range(5):
                    event = "split_{}".format(split)
                    print(event)

                    root_dir_train = "processed_twitter16_{}_svm".format(train_lang)
                    root_dir_test = "processed_twitter16_{}_svm".format(test_lang)
                    with open(os.path.join(root_dir_test, "twitter16_split_{}_test_content".format(split)),
                              "rb") as handle:
                        X_test = pickle.load(handle)
                    with open(os.path.join(root_dir_test, "twitter16_split_{}_test_label".format(split)),
                              "rb") as handle:
                        y_test_raw = pickle.load(handle)

                    # X_train = []
                    # y_train_raw = []

                    with open(os.path.join(root_dir_test, "twitter16_split_{}_train_content".format(split)),
                              "rb") as handle:
                        X_train = pickle.load(handle)
                    with open(os.path.join(root_dir_test, "twitter16_split_{}_train_label".format(split)),
                              "rb") as handle:
                        y_train_raw = pickle.load(handle)

                    y_test_raw = [int(x) for x in y_test_raw]
                    y_train_raw = [int(x) for x in y_train_raw]

                    print(len(X_train))
                    print(X_test[0])
                    print(len(y_train_raw))
                    print(np.unique(y_train_raw))
                # for train_index, test_index in logo.split(X, y, groups):
                #     #     print("TRAIN:", train_index, "TEST:", test_index)
                #     X_train, X_test = X[train_index].tolist(), X[test_index].tolist()
                #     y_train, y_test = y[train_index].tolist(), y[test_index].tolist()
                #
                #     X_train_processed = [preprocess_en_text(i) for i in X_train]
                #     X_test_processed = [preprocess_en_text(i) for i in X_test]


                    # for i in range(len(X_train)):
                    #     X_train[i] = re.sub(
                    #         r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
                    #         "<url>", X_train[i]).strip()
                    # for i in range(len(X_test)):
                    #     X_train[i] = re.sub(
                    #         r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
                    #         "<url>", X_test[i]).strip()

                    # print(X_train, X_test, y_train, y_test)
                    # print(X_train)
                    y_test = y_test_raw
                    y_train = y_train_raw

                    vectorizer = TfidfVectorizer(stop_words='english',
                                                 max_features=num_features,
                                                 sublinear_tf=True,
                                                 use_idf=True)
                    train_vectors = vectorizer.fit_transform(X_train)
                    test_vectors = vectorizer.transform(X_test)

                    svm_classifier = svm.SVC(kernel='rbf', class_weight="balanced", random_state=1111, C=C_value, gamma=gamma_value)
                    svm_classifier.fit(train_vectors, y_train)
                    prediction = svm_classifier.predict(test_vectors)

                    report = classification_report(y_test, prediction, output_dict=True, zero_division=1)
                    results.append(report)
                    print(report)

                    print(prediction)
                    print(y_test)
                    print(len(prediction))
                    print(len(y_test))

                    raw_results.append({"ground_truth": y_test, "preds": prediction})

                results_save_dir = "svm_twitter16_results_{}_{}".format(train_lang, test_lang)
                if not os.path.exists(results_save_dir):
                    os.mkdir(results_save_dir)

                summary = {"run{}".format(i+1): results[i] for i in range(len(results))}
                # with open("svm_mixed_test_new/svm_test_results_{}_{}_{}_{}.json".format(train_lang, num_features, C_value, gamma_value), 'w') as f:
                #     json.dump(summary, f, indent=4)
                with open(os.path.join(results_save_dir, "raw_results_{}_{}_{}.pickle".format(num_features, C_value, gamma_value)), "wb") as ff:
                    pickle.dump(raw_results, ff)