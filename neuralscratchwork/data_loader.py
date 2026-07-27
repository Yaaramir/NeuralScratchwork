class Data_Loader():

    def unpickle(file):
        import pickle
        with open(file, 'rb') as data:
            dict = pickle.load(data, encoding='bytes')
        return dict