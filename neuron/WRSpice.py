import numpy as np
import os
import fileinput
import sys, os, scipy


class WRSpice:
    # Files are without extentions
    def __init__(self, tempCirFile, OutDatFile, pathWRSSpice='', pathCir='',stepTran='1n', stopTran='10n'):
        self.tempCirFile = tempCirFile
        self.OutDatFile = OutDatFile
        self.pathWRSSpice = pathWRSSpice
        self.pathCir = pathCir
        self.stepTran = stepTran
        self.stopTran = stopTran
        self.Params = None
        self.cirString=[]
        self.FilePrefix='0'
        self.save=''
    def templateFileRead(self):
        self.cirString=[]
        with open(self.tempCirFile +'.cir') as file_in:
            for line in file_in:
                self.cirString.append(line)
        

    def ReplaceParams(self):
        if self.Params is None:
            return
        for k, v in self.Params.items():       
            tempString=[]
            for line in self.cirString:
                ll = line.split(' ')
                if(ll[0] == k):
                    line = ll[0]+' '+ll[1]+' '+ll[2]+' '+v+'\n'
                elif line.find(k)>-1 and line.find('K')==-1:
                    temp=''
                    for i in ll:
                        if(i == k):
                            i=v
                        temp=temp +' ' +i
#                        temp=temp +'\n'
                    line=temp
                    
                tempString.append(line)               
                self.cirString=tempString

            
    def PutInitialConditions(self):
        data_dict = self.read_wr_data()   
        self.templateFileRead()
        self.ReplaceParams()
        tempString=[]
        for line in self.cirString:
            line=line.strip()
            ll = line.split(' ')
            if(ll[0]).find('B')>-1:
                node=ll[1]
                temp=data_dict['v('+node+')']
                vj=temp[len(temp)-1]
 
                node=ll[3]
                temp=data_dict['v('+node+')']
                phi=temp[len(temp)-1]
                
                line=line.strip()+' ic='+str(vj)+','+str(phi)+'\n'

            elif(ll[0]).find('L')>-1:
                temp=data_dict[ll[0]+'#branch']
                i=temp[len(temp)-1]

                line=line.strip()+' ic='+str(i)+'\n'
                
            elif(ll[0] == 'V0'):
                line='V0 1 0 pwl(0 0 0.02ns 0 0.03ns 0)\n'
            elif(ll[0]).find('I')>-1:
                if line.find('pwl')>-1:
                    s=ll[8]
                    line=ll[0]+' '+ll[1]+' '+ll[2]+' '+ll[3]+' '+s[0:len(s)-1]+' '+ll[5]+' '+s[0:len(s)-1]+' '+ll[7]+' '+ll[8]
#                if line.find('pulse')>-1:
                    
#                    line=ll[0]+' '+ll[1]+' '+ll[2]+' '+ll[3]+' '+ll[5]+' '+ll[5]+' '+ll[6]+' '+ll[7]
                
                line=line+'\n'
            else:
                line=line+'\n'
                
            tempString.append(line)               
            self.cirString=tempString
            
    def cirFileSave(self):
        strControl=''
        print(self.OutDatFile+ self.FilePrefix+ '.cir')
        
        strControl=strControl+'.control\n' 
        strControl=strControl+'set maxdata=2560000\n' 
#        strControl=strControl+'tran '+self.stepTran+' '+self.stopTran +' uic\n'
        strControl=strControl+'tran '+self.stepTran+' '+self.stopTran +'\n'
#        strControl=strControl+'plot  i(l5) v(6)\n'
        strControl=strControl+'write '+'./'+self.pathCir+self.OutDatFile+ self.FilePrefix+ '.dat '+self.save+'\n'
        strControl=strControl+'quit\n'
        strControl=strControl+'.endc\n'
        
        f_Out = open(self.pathCir+self.OutDatFile+ self.FilePrefix+ '.cir', 'w')
        f_Out.write(''.join(self.cirString)+strControl)
        f_Out.close()

        
    def run(self):
        cirFile=self.OutDatFile+ self.FilePrefix+ '.cir'
        os.system(self.pathWRSSpice +' '+'./'+self.pathCir+ cirFile)
        
    def runInSteps(self,No):
        stopTran=self.stopTran
        vTran=stopTran[0:len(stopTran)-1]
        uTran=stopTran[len(stopTran)-1:len(stopTran)-0]
        step=float(vTran)/No
        
        self.templateFileRead()
        self.ReplaceParams()
        
        self.stopTran = str(step)+uTran
        self.FilePrefix='0'
        self.cirFileSave()
        self.run()
        
        for i in np.arange(1,No,1):
            print('\niteration:'+str(i))
            self.PutInitialConditions()
            self.FilePrefix=str(i)
            self.cirFileSave()
            
            self.run()
    def doAll(self):
        self.templateFileRead()
        self.ReplaceParams()
        self.cirFileSave()
        self.run()
        
    def read_wr_data(self):
        print('reading wr data file ...')
        f = open('./'+self.pathCir+self.OutDatFile+ self.FilePrefix+ '.dat', 'rt')
    
        file_lines = f.readlines()
    
        counter = 0
        for line in file_lines:
            counter += 1
            if line.find('No. Variables:') != -1:
                ind_start = line.find('No. Variables:')
                num_vars = int(line[ind_start+15:])
            if line.find('No. Points:') != -1:
                ind_start = line.find('No. Points:')
                num_pts = int(line[ind_start+11:])
            if str(line) == 'Variables:\n':            
                break    

        var_list = []
        for jj in range(num_vars):
            if jj <= 9:
                var_list.append(file_lines[counter+jj][3:-3]) 
            if jj > 9:
                var_list.append(file_lines[counter+jj][4:-3]) 
    
        data_mat = np.zeros([num_pts,num_vars])
        tn = counter+num_vars+1
        for ii in range(num_pts):
            # print('\n\nii = {}\n'.format(ii))
            for jj in range(num_vars):
                ind_start = file_lines[tn+jj].find('\t')
                # print('tn+jj = {}'.format(tn+jj))
                data_mat[ii,jj] = float(file_lines[tn+jj][ind_start+1:])
                # print('data_mat[ii,jj] = {}'.format(data_mat[ii,jj]))
            tn += num_vars
        
        f.close
        
        data_dict = dict()
        for ii in range(num_vars):
            data_dict[var_list[ii]] = data_mat[:,ii]
            
        print('done reading wr data file.')
        
        return data_dict
    
