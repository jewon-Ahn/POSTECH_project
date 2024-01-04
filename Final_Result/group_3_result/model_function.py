import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import tqdm
import re
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
import matplotlib.font_manager as fm

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

def create_folder_if_not_exists(folder_path):
    """
    주어진 폴더 경로에 폴더가 없을 경우 폴더를 생성합니다.
    
    Parameters:
        folder_path (str): 생성할 폴더의 경로
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"폴더 생성: {folder_path}")


def categorical_encoder_def(df, categorical_column):
    encoder_list = []
    for i in categorical_column:
        label_encoder = LabelEncoder()
        df[i] = label_encoder.fit_transform(df[i])
        encoder_list.append(label_encoder)

    mapping_list = []
    for encoder in encoder_list:
        mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
        mapping_list.append(mapping)

    return df, encoder_list, mapping_list

def ordinal_encoder_def(df, ordinal_column, ordinal):

    ordinal_encoder = OrdinalEncoder(categories=ordinal)
    df[ordinal_column] = ordinal_encoder.fit_transform(df[ordinal_column])

    ordinal_mapping_list = [dict(zip(i,range(len(i)))) for i in ordinal_encoder.categories_]

    return df, ordinal_encoder, ordinal_mapping_list

def preprocess(data,categorical_column,ordinal_column,ordinal_order):

    #categorical
    data, categorical_encoder_list, categorical_mapping_list = categorical_encoder_def(data,categorical_column)
    #ordinal
    data, ordinal_encoder, ordinal_mapping_list = ordinal_encoder_def(data,ordinal_column,ordinal_order)
    #ordinal encoder에서 nan을 계속 nan으로  함 그래서 0으로 채움,categorical은 na를 알아서 encoding 시켜줌
    data = data.fillna(0)

    return data,categorical_encoder_list,categorical_mapping_list,ordinal_encoder, ordinal_mapping_list


class model_use():
    def __init__(self, model, batch_size, epochs, want_criterion, device, num_cat, save_path):
        self.model = model.to(device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = want_criterion
        self.device = device
        self.num_cat = num_cat
        self.save_path = save_path

    def test(self,valid_x,valid_y):
        X_valid = torch.tensor(valid_x.values, dtype=torch.long)
        Y_valid = torch.tensor(valid_y.values, dtype=torch.long)

        num_of_col = valid_x.shape[1]

        self.model.eval()
        with torch.no_grad():
            category_valid_X = X_valid[:,:self.num_cat]
            cont_valid_X = torch.tensor([[]]) 
            if num_of_col != self.num_cat:
                    cont_valid_X = torch.tensor(X_valid[:,self.num_cat:],dtype=torch.float)

            category_valid_X, cont_valid_X, Y_valid = category_valid_X.to(self.device), cont_valid_X.to(self.device), Y_valid.to(self.device)
            outputs, attns, embedding, before_x = self.model(category_valid_X,cont_valid_X,return_attn=True)

            outputs = torch.squeeze(outputs)
            Y_valid = Y_valid.to(torch.float)
            
            #test_loss = self.criterion(outputs,Y_valid)
            predict = (outputs>0.5).to(torch.int)
            #cuda 
            Y_valid = Y_valid.to(torch.int)
            true_labels = Y_valid.cpu().numpy()
            predicted_labels = predict.cpu().numpy()

            accuracy = accuracy_score(true_labels, predicted_labels)
            recall = recall_score(true_labels,predicted_labels)
            f1 = f1_score(true_labels, predicted_labels)

        return embedding, accuracy, recall, f1


    def train(self,train_x,train_y,valid_x,valid_y):
        X_train = torch.tensor(train_x.values, dtype=torch.long)
        Y_train = torch.tensor(train_y.values, dtype=torch.long)

        num_of_col = X_train.shape[1]

        my_dataset = TensorDataset(X_train,Y_train)
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        my_dataloader = DataLoader(my_dataset, batch_size=self.batch_size, shuffle=True)

        total_train_loss = []

        for epoch in range(0,self.epochs):
            train_loss_list = []

            self.model.train()
            for batch_X, batch_y in tqdm.tqdm(my_dataloader):
                category_batch_X = batch_X[:,:self.num_cat]
                cont_batch_X = torch.tensor([[]]) 
                if num_of_col != self.num_cat:
                    cont_batch_X = torch.tensor(batch_X[:,self.num_cat:],dtype=torch.float)
                
                category_batch_X, cont_batch_X, batch_y = category_batch_X.to(self.device), cont_batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.model(category_batch_X,cont_batch_X,return_attn=False)
                output = torch.squeeze(output)
                batch_y = batch_y.to(torch.float)

                loss = self.criterion(output,batch_y)
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss)

            batch_average_loss = sum(train_loss_list)/len(my_dataloader)
            total_train_loss.append(batch_average_loss)

            embedding,accuracy,recall,f1 = self.test(valid_x,valid_y)

            print(f"epoch: {epoch} | train_loss: {batch_average_loss:.4f}")
            print(f"accuracy: {accuracy:.2f} | f1-score: {f1:.2f} | recall: {recall:.2f}")
            print("-----------------------------------------------------------------")

            create_folder_if_not_exists(self.save_path)

            model_name = f"/epoch={epoch},f1={f1:.2f},recall={recall:.2f}.pth"
            torch.save(self.model.state_dict(),self.save_path+model_name)
    
    def uncertainty(self,iteration,valid_x,valid_y):
        X_valid = torch.tensor(valid_x.values, dtype=torch.long)
        Y_valid = torch.tensor(valid_y.values, dtype=torch.long)

        num_of_col = valid_x.shape[1]
        output_array = []
        self.model.train()
        for _ in tqdm.tqdm(range(iteration)):
            with torch.no_grad():
                category_valid_X = X_valid[:,:self.num_cat]
                cont_valid_X = torch.tensor([[]]) 
                if num_of_col != self.num_cat:
                        cont_valid_X = torch.tensor(X_valid[:,self.num_cat:],dtype=torch.float)

                category_valid_X, cont_valid_X, Y_valid = category_valid_X.to(self.device), cont_valid_X.to(self.device), Y_valid.to(self.device)
                outputs, attns, embedding, before_x = self.model(category_valid_X,cont_valid_X,return_attn=True)

                outputs = torch.squeeze(outputs)
                outputs = outputs.cpu()

                output_array.append(outputs)

        outputs_array = []

        for i in output_array:
            outputs_array.append(i.numpy())

        #평균 및 분산
        mean_array = np.mean(outputs_array,axis=0)
        var_array = np.std(outputs_array,axis=0)

        #분산의 평균
        total_uncertainty = np.mean(var_array)

        predict = (mean_array>0.5).astype(int)

        true_labels = Y_valid.cpu().numpy()

        accuracy = accuracy_score(true_labels, predict)
        recall = recall_score(true_labels,predict)
        f1 = f1_score(true_labels, predict)

        print(f"accuracy: {accuracy:.2f} | f1-score: {f1:.2f} | recall: {recall:.2f}")

        return mean_array,total_uncertainty
    
    def embedding_graph(self,X,Y,save_name = None):
        num_of_col = X.shape[1]
        X_total = torch.tensor(X.values, dtype=torch.long)
        Y_total = torch.tensor(Y.values, dtype=torch.long)

        my_dataset = TensorDataset(X_total,Y_total)

        my_dataloader = DataLoader(my_dataset, batch_size=self.batch_size, shuffle=True)

        a = pd.DataFrame()
        b = np.array([])
        c = np.array([])

        for batch_X, batch_y in tqdm.tqdm(my_dataloader):
            X_valid,Y_valid = batch_X.to(self.device), batch_y.to(self.device)

            category_valid_X = X_valid[:,:self.num_cat]
            cont_valid_X = torch.tensor([[]]) 
            if num_of_col != self.num_cat:
                    cont_valid_X = torch.tensor(X_valid[:,self.num_cat:],dtype=torch.float)

            outputs, attns, embedding, before_x = self.model(category_valid_X ,cont_valid_X,return_attn=True)
            cpu_embedding = embedding.cpu().detach()
            df_embedding = pd.DataFrame(cpu_embedding.numpy())

            predict = (outputs>0.5).to(torch.int)

            a = pd.concat([a,df_embedding])
            b = np.append(b,Y_valid.cpu().numpy())
            c = np.append(c,predict.cpu().numpy())

        print("To do TSNE takes time. Please wait")
        tsne_np = TSNE(n_components = 2).fit_transform(a)
        df_a = pd.DataFrame(tsne_np)
        df_a["label"] = b
        df_a["predict"] = c

        plt.title("Embedding by transformer")
        plt.scatter(df_a[df_a["label"]==0][0],df_a[df_a["label"]==0][1],label="not buy",alpha=0.2)
        plt.scatter(df_a[df_a["label"]==1][0],df_a[df_a["label"]==1][1],label="buy",alpha=0.2)
        plt.legend()

        if save_name != None:
            print(f"figrue is saved in {self.save_path}")
            plt.savefig(self.save_path+save_name)

        plt.show()

    def column_embedding_graph(self,X,Y,all_col,categorical_column,categorical_mapping_list,ordinal_column,ordinal_mapping_list,save_name = None):
        num_of_col = X.shape[1]
        X_total = torch.tensor(X.values, dtype=torch.long)
        Y_total = torch.tensor(Y.values, dtype=torch.long)

        my_dataset = TensorDataset(X_total,Y_total)

        my_dataloader = DataLoader(my_dataset, batch_size=self.batch_size, shuffle=False)

        total_embedding = pd.DataFrame()
        embedding_df = pd.DataFrame()


        for batch_X, batch_y in tqdm.tqdm(my_dataloader):
            X_valid,Y_valid = batch_X.to(self.device), batch_y.to(self.device)

            category_valid_X = X_valid[:,:self.num_cat]
            cont_valid_X = torch.tensor([[]]) 
            if num_of_col != self.num_cat:
                    cont_valid_X = torch.tensor(X_valid[:,self.num_cat:],dtype=torch.float)

            outputs, attns, embedding, before_x = self.model(category_valid_X ,cont_valid_X,return_attn=True)
            cpu_embedding = embedding.cpu().detach()
            df_embedding = pd.DataFrame(cpu_embedding.numpy())

            total_embedding = pd.concat([total_embedding,df_embedding])

            
        for i in tqdm.tqdm(range(0,len(all_col))):
            selected_col = all_col[i]
            all_values = X[selected_col].unique()

            if selected_col in ["거주시군구명","거주행정동명"]:
                continue

            #key-value 전환
            if selected_col in categorical_column:
                dict_index = categorical_column.index(selected_col)
                selected_mapping_list = categorical_mapping_list[dict_index]
            else:
                dict_index = ordinal_column.index(selected_col)
                selected_mapping_list = ordinal_mapping_list[dict_index]

            for v in all_values:
                selected_index = X[X[selected_col]==v].index[0]
                index = [key for key, value in selected_mapping_list.items() if value == v][0]
                col_name = selected_col + str(index)

                embedding_value = total_embedding.iloc[selected_index,32*i:32*(i+1)]
                temp_df = pd.DataFrame({col_name : embedding_value.values.tolist()})
                embedding_df = pd.concat([embedding_df,temp_df],axis=1)

        embedding_df = embedding_df.T
        print("To do TSNE takes time. Please wait")
        tsne_np_column = TSNE(n_components = 2).fit_transform(embedding_df)
        column_tsne_df = pd.DataFrame(tsne_np_column)
        column_tsne_df.index =embedding_df.index

        font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕 폰트 경로
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()

        # Seaborn 설정
        sns.set(font=font_prop.get_name())

        plt.figure(figsize=(15,10))
        sns.scatterplot(column_tsne_df, x =0,y=1, hue = column_tsne_df.index)
        plt.legend(bbox_to_anchor=(1, 1))
        if save_name != None:
            print(f"figrue is saved in {self.save_path}")
            plt.savefig(self.save_path+save_name)
        plt.show()



