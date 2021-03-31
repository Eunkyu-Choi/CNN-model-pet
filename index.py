from image_classifier import ImageClassifier

if __name__ == '__main__':
    # 반려동물의 종 입력(dog, cat, etc)
    Type = input('Type of Pet :')

    if Type == 'dog' or Type == 'cat':
        model_path = 'saved_model_{}/my_model'.format(Type)
        f = open('class_list_{}.txt'.format(Type), 'r')
        # '''Path of Image to Classification'''
        ic = ImageClassifier(model_path, f)
        path = input('Path of Image to Classify Breed :')
        ic.classify(path)

    else:
        # 개나 고양이가 아닐경우 직접 입력
        breed = input('Breed of Pet :')
        print(breed)
