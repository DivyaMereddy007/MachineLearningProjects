class Test:
    def __init__(self,a,b):
        self.a=a
        self.b=b

    def add(self):
        return self.a+self.b

    def sub(self):
        return self.a - self.b

    def mult(self):
        return self.a * self.b

    def div(self):
        return self.a / self.b

    def user_time_series_train_test_split(self,which_dates):
        for date_ in which_dates:
            rnd_check = 0.0
        #             if rnd.random()<rnd_check:
        #                 continue
            self.date_ = date_
            self.start_t_train = time.time()
            print('\n====== Date = {} ====== '.format(date_) )
            ################### test ###################
            self.df_test, self.test_cols = self.readSpotTemplate(date_ = date_)
            self.df_test = self.defaultRates(self.df_test)
            ################### train ###################
            self.df_train = self.df_loads.drop(self.df_loads[self.df_loads.BookByDateTime_N >= date_].index)
            self.df_train = self.df_train[(self.df_train.BookByDateTime_N >= self.date_-self.days )\
                |  (self.df_train.BookByDateTime_N.isin(np.arange(self.date_-1*365-self.day_b ,self.date_-1*365+self.day_a)) ) \
                |  (self.df_train.BookByDateTime_N.isin(np.arange(self.date_-2*365-self.day_b ,self.date_-2*365+self.day_a)) ) ]#\
            self.train_model()
            self.predict_()
            self.save_results()

x=1;y=2
v1=Test(1,2)
v2=Test(1,3)

v1.add()
v2.add()
v1+v2
