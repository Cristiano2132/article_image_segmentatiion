from sklearn.model_selection import train_test_split
from .my_confusion_matrix import My_Confusion_Matrix
from sklearn.metrics import confusion_matrix
from .models import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import time

class ModelsTraining:
    def __init__(self, data_path:str, save_folder_path:str, get_indices=False) -> None:
        self.__data_path=data_path
        self.__save_folder_path = save_folder_path
        self.__get_indices = get_indices
        self.random_state = 1
        
    def trainin_models(self):
        
        X, y = structurer_data(path_data=self.__data_path, indices=self.__get_indices)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=self.random_state)
        del (X, y)
        print('Iniciando ajuste dos modelos!!!')
        list_models = ['logit', 'rf', 'xgb']
        dict_times = {}
        for model in list_models:
            print(f'\n\n#########  Rodando o modelo: {model} ##########')
            factory = Factory(classificator_name=model, X=X_train, y=y_train, save_folder_path=self.__save_folder_path)
            start_time = time.time()
            factory.create_regressor().fit_regression()
            time_during = time.time() - start_time
            dict_times[model]=time_during
            print(f"Optimization time --- {time_during} seconds ---")

        keys = dict_times.keys()
        values = dict_times.values()
        df = pd.DataFrame(columns=['model', 'time'])
        df['model'] = ['RL', 'RF', 'XGBoost']
        df['time'] = values
        # Figure Size
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.yticks(fontsize=18)
        # Horizontal Bar Plot
        ax.barh(df.model, df.time)
        # Add annotation to bars
        for i in ax.patches:
            plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                    f'{i.get_width():.2f}',
                    fontsize=18, fontweight='bold',
                    color='grey')

        # Add Plot Title
        ax.set_title('Time spent in the model fitting process',
                    loc='left')
        plt.xlabel("Time(seconds)")
        plt.tight_layout()
        plt.savefig(f'{self.__save_folder_path}/spend_time.png', dpi=100)

        path_xgb_model = f'{self.__save_folder_path}/xgb_model.pkl'
        path_rf_model = f'{self.__save_folder_path}/rf_model.pkl'
        path_logit_model = f'{self.__save_folder_path}/logit_model.pkl'
        logit_model_loaded = pickle.load(open(path_logit_model, "rb"))
        xgb_model_loaded = pickle.load(open(path_xgb_model, "rb"))
        rf_model_loaded = pickle.load(open(path_rf_model, "rb"))
        models_list = [logit_model_loaded, xgb_model_loaded, rf_model_loaded]
        model_name = ['logit', 'xgb', 'rf']

        for n,model in enumerate(models_list):
            y_pred = model.predict(X_test)
            cf_matrix = confusion_matrix(y_test, y_pred)
            title = f'Confusion Matrix - Modelo {model_name[n]}'
            group_names = ['TrueNeg', 'FalsePos', 'FalseNeg', 'TruePos']
            cm = My_Confusion_Matrix(cf=cf_matrix,
                                    title=title,
                                    group_names=group_names,
                                    cmap='Blues',
                                    figsize=(10, 12),
                                    categories=['Solo', 'Vegetaçāo'],
                                    save_path=f'{self.__save_folder_path}/confusion_matrix_{model_name[n]}.png'
                                    )
            cm.make_confusion_matrix()

        df = get_acuracy(models_list, model_name, X_test, y_test)
        # print(df)
        return df



def structurer_data(path_data: str, indices=False):
    df = pd.read_csv(path_data, index_col=0)
    dic_replace = {'Vegetacao': 0, 'Solo': 1}
    df['target'] = df['target'].replace(dic_replace)
    df = df.set_index(['x', 'y']).dropna()
    df = df.assign(MGVRI=lambda x: (x.green ** 2 - x.red ** 2) / (x.green ** 2 + x.red ** 2))
    df = df.assign(GLI=lambda x: (2 * x.green - x.red - x.blue) / (2 * x.green + x.red + x.blue))
    df = df.assign(MPRI=lambda x: (x.green - x.red) / (x.green + x.red))
    df = df.assign(RGVBI=lambda x: (x.green - x.red * x.blue) / (x.green ** 2 + x.red * x.blue))
    df = df.assign(ExG=lambda x: (2 * x.green - x.red - x.blue) / (x.green + x.red + x.blue))
    df = df.assign(VEG=lambda x: (x.green) / (x.red ** 0.667 + x.blue ** (1 - 0.667)))
    df = df.dropna()
    y = df['target']
    if indices:
        X = df[['MGVRI', 'GLI', 'MPRI', 'RGVBI', 'ExG', 'VEG']]
    else:
        X = df[['red', 'blue', 'green']]
    return X, y

def get_acuracy(models_list, model_name, X_test, y_test):
    df_results = pd.DataFrame()
    for n,model in enumerate(models_list):
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose().reset_index()
        df['model']=model_name[n]
        # mask = df.index== 0 and df.index == 1 and df.index == 'acuracy'
        df_results = df_results.append(df, ignore_index=True)
    df_results = df_results[['model', 'index', 'precision', 'recall', 'f1-score']]
    print(df_results)
    return df_results

        


    


