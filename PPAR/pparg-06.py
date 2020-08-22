
#import logging
#logging.getLogger('tensorflow').disabled = True

import sys
import numpy as np
import random
from keras import layers, models
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from rdkit import rdBase
from rdkit.Chem import PropertyMol
# from rdkit.Chem.Draw import IPythonConsole

#tf.get_logger().setLevel('INFO')
#tf.autograph.set_verbosity(1)



def splitTrainTestSet( molsdf, frac_test ) :
        moles = [ m for m in molsdf if m != None ]
        # random.shuffle( moles )
        # moles_train = moles[ :70 ]
        # moles_test = moles[ 70: ]
        moles_train, moles_test = train_test_split( moles, test_size=frac_test )
        # print( f'Train/Test = {len(moles_train)} / {len(moles_test)}' )
        return moles_train, moles_test

def readMolecules( fname ) :
        return Chem.rdmolfiles.SDMolSupplier( fname )

def readAndSplitMolecules( fname, frac_test ) :
        molsdf = readMolecules( fname )
        print( fname, ' : ', f'Number of molecules = {len( molsdf )}' )
        return splitTrainTestSet( molsdf, frac_test )


def assignFingerprint( moles, nBits=2048 ) :
        df = pd.DataFrame( columns=[ 'name', 'pChEMBL_Value', 'fingerprint', 'activity_calc', 'activity_diff' ] )
        for m in moles:
                name = m.GetProp( 'Molecule ChEMBL ID' )
                activity = float( m.GetProp( 'pChEMBL_Value' ) )
                fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect( m, radius=2, nBits=nBits )
                #arr = np.zeros( (1, ) )
                #DataStructs.cDataStructs.ConvertToNumpyArray( fp, arr )
                df = df.append( { 'name' : name, 'pChEMBL_Value' : activity, 'fingerprint' : fp },
                        ignore_index=True )
        df.reset_index( inplace=True )
        return df


def setupFeatureDataFrame( molsdf, frac_test=0.3, nBits=2048 ) :
        moles = [ m for m in molsdf if m != None ]
        moles_train, moles_valid = train_test_split( moles, test_size=frac_test )
        df_train = assignFingerprint( moles_train, nBits )
        df_valid = assignFingerprint( moles_valid, nBits )
        return df_train, df_valid


def splitTrainTest( df, frac_test=0.3 ) :
        count = df.shape[0]
        idx = np.array( np.arange(count) )
        idA, idB = train_test_split( idx, test_size=frac_test )
        df.loc[ idA, 'set' ] = 0
        df.loc[ idB, 'set' ] = 1


def getFeatureMatrix( fps ) :
        nps = []
        for fp in fps :
                arr = np.zeros( (1, ) )
                DataStructs.cDataStructs.ConvertToNumpyArray( fp, arr )
                nps.append( arr )
        return np.array( nps )


def getFeatures( df ) :
        #x_train = getFeatureMatrix( df[ df.set != 1 ].fingerprint )
        #y_train = df[ df.set != 1 ].pChEMBL_Value.to_numpy()
        #x_valid = getFeatureMatrix( df[ df.set == 1 ].fingerprint )
        #y_valid = ( df[ df.set == 1 ].pChEMBL_Value ).to_numpy()

        pos = df.loc[ :, 'set' ] == 1
        #pos_v = df[pos].index
        #pos_t = df[ ~pos].index
        #print( pos_t )
        #print( pos_v )
        #sys.exit()

        x_train = getFeatureMatrix( df[ ~ pos ].fingerprint )
        y_train = df[ ~pos ].pChEMBL_Value.to_numpy()
        x_valid = getFeatureMatrix( df[ pos ].fingerprint )
        y_valid = ( df[ pos ].pChEMBL_Value ).to_numpy()
        return x_train, y_train, x_valid, y_valid




def getFeature( df ) :
        x = getFeatureMatrix( df.fingerprint )
        y = ( df.pChEMBL_Value ).to_numpy()
        return x, y




def getFingerprintFromMolecule( moles, nBits=2048 ) :
        fps = [ Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect( m, radius=2, nBits=nBits ) for m in moles ]
        np_fps = []
        for fp in fps:
                arr = np.zeros( (1, ) )
                DataStructs.cDataStructs.ConvertToNumpyArray( fp, arr )
                np_fps.append( arr )
        return np_fps


def classify_activity( y, criteria ) :
        y = y > criteria
        # print( f'Num true/false = {(y==True).sum()} / {(y==False).sum()}' )
        y = np_utils.to_categorical( y, 2 ).astype( 'bool' )
        return y


def getActivityOfMolecule( moles ) :
        try:
                activity = [ m.GetProp( 'pChEMBL_Value' ) for m in moles ]
                activity = np.asarray( activity ).astype( 'float' )
        except :
                print( "No activity data..." )
                activity = np.array( len(moles) )
        return activity


def getClassFromActivity( moles, criteria ) :
        activity = getActivityOfMolecule( moles )
        return classify_activity( activity, criteria )


def countTrue( y ) :
        return y.sum( axis=0 )[1]


def generateInputDataClass( moles, criteria, label, nBits=2048 ) :
        x = np.array( getFingerprintFromMolecule( moles, nBits ) )
        y = getClassFromActivity( moles, criteria )
        print( f'{label} = {len(y)} = {countTrue(y)} + {len(y) - countTrue(y)}' )
        return x, y

def generateInputDataRegres( moles, nBits=2048 ) :
        x = np.array( getFingerprintFromMolecule( moles, nBits ) )
        y = getActivityOfMolecule( moles )
        return x, y


class DNN( models.Sequential):
        def __init__( self, Nin, Nh_1, Nout ) :
                super().__init__()
                self.add( layers.Dense( Nh_1[0], activation = 'relu', input_shape = (Nin,), name='Hidden-1' ) )
                self.add( layers.Dense( Nh_1[1], activation = 'relu', input_shape = (Nin,), name='Hidden-2' ) )
                self.add( layers.Dense( Nout, activation = 'softmax' ) )
                self.compile( loss='categorical_crossentropy', optimizer='adam',metrics=[ 'accuracy' ] )


def plot_loss( history ) :
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc=0)

def plot_acc( history ) :
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc=0)



def do_classification( x_train, y_train, x_test, y_test ):
        number_of_class = 2
        Nin = x_train.shape[1]
        Nh_1 = [ Nin, Nin, Nin/2 ]

        model = DNN( Nin, Nh_1, number_of_class )
        history = model.fit( x_train, y_train, epochs=100, batch_size=100, validation_split=0.3, verbose=2)
        performance_test = model.evaluate( x_test, y_test, batch_size=100)
        print( 'Test Loss and Accuracy ->', performance_test )

        plot_loss( history )
        # plot_acc( history )
        plt.show()





def make_regression_model( X_train, Y_train, X_validation, Y_validation, epochs=100, batch=10 ) :
        nfeatures = X_train.shape[1]
        model = Sequential()
        model.add(Dense(nfeatures, input_dim=nfeatures, activation='relu'))
        model.add(Dense(nfeatures, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'acc'] )
        hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch,
                validation_data=(X_validation, Y_validation), verbose=2)
        return model, hist


def plot_regression( model, hist, X_train, Y_train, X_validation, Y_validation ) :
        YT_prediction = model.predict(X_train).flatten()
        YV_prediction = model.predict(X_validation).flatten()

        np.set_printoptions(threshold=sys.maxsize)
        sys.stdout = open( 'qsar-validation.txt', 'w' )
        print( np.c_[ Y_train, YT_prediction ] )
        print( np.c_[ Y_validation, YV_prediction ] )


        plt.figure( figsize=(4, 4) )
        plt.scatter(Y_train, YT_prediction, color='black', s=2)
        plt.scatter(Y_validation, YV_prediction, color='red', s=3)
        plt.xticks( np.arange(3, 12) )
        plt.yticks( np.arange(3, 12) )
        # plt.show()
        # plt.figure( figsize=(4, 4) )
        plt.savefig( 'regression-xy.png', dpi=300 )


        plt.figure( figsize=(6, 4) )
        _, loss_ax = plt.subplots()
        _ = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc='upper right')

        #acc_ax.plot(hist.history['acc'], 'b', label='train acc')
        #acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
        #acc_ax.set_ylabel('accuracy')
        #acc_ax.legend(loc='upper left')

        plt.show()
        plt.savefig( 'regression-history.png', dpi=300 )

        return



# delete outliers
#       dels = np.array( [ 644 , 512 , 109 , 38 , 494 , 520 , 683 , 2  , 289 , 574 , 543 , 153 , 125 , 679, 758 , 91 , 384 ] )#
#       X_validation = np.delete( X_validation, dels, axis=0 )
#       Y_validation = np.delete( Y_validation, dels )

def do_regression( X_train, Y_train, X_validation, Y_validation ):
        model, hist = make_regression_model( X_train, Y_train, X_validation, Y_validation )
        plot_regression( model, hist, X_train, Y_train, X_validation, Y_validation )
        return model, hist



def do_prediction( model, fname ) :
        molsdf = Chem.rdmolfiles.SDMolSupplier( fname )
        name = [ m.GetProp( 'Name' ) for m in molsdf ]

        x_test,  y_test  = generateInputDataRegres( molsdf )
        y_test = model.predict( x_test ).flatten()
        sys.stdout = open( 'qsar-biocide.txt', 'w' )
        print( np.c_[ name, y_test ] )



def run_classification( sdfname, frac_test, nBits, criteria ) :
        sdfname = "./200303-chembl4qsar_6101.sdf"
        moles_train, moles_valid = readAndSplitMolecules( sdfname, frac_test )
        x_train, y_train = generateInputDataClass( moles_train, criteria, 'Training', nBits )
        x_valid, y_valid = generateInputDataClass( moles_valid,  criteria, 'Test ', nBits )
        do_classification( x_train, y_train, x_valid, y_valid )


def run_regression( sdfname, frac_test, nBits, fname_model ) :
        sdfname = "./200609-ChEMBL-2672.sdf"
        moles_train, moles_valid = readAndSplitMolecules( sdfname, frac_test )
        x_train, y_train = generateInputDataRegres( moles_train, nBits )
        x_valid, y_valid  = generateInputDataRegres( moles_valid, nBits )
        model, _ = do_regression( x_train, y_train, x_valid, y_valid )
        model.save( fname_model )






def  adjustTrainTestSet_0( df_train, df_valid, model, X_valid, Y_valid ) :
        YV_prediction = model.predict(X_valid).flatten()
        df_valid.activity_calc = YV_prediction
        df_valid.activity_diff = abs( df_valid.pChEMBL_Value - df_valid.activity_calc )
        # diff = abs( Y_valid - YV_prediction )
        diff = df_valid.activity_diff.to_numpy()
        maxk = np.argmax( diff )
        row = df_valid.iloc[maxk]
        print( '    moving... ', row.name, ', ', row.pChEMBL_Value, ', ', diff[maxk] )
        df_train = df_train.append( row, ignore_index=True )
        df_valid.drop( maxk, inplace=True )
        return df_train, df_valid



def  adjustTrainTestSet( df_train, df_valid, model, X_valid, Y_valid, crit=1.0 ) :
        YV_prediction = model.predict(X_valid).flatten()
        df_valid.activity_calc = YV_prediction
        df_valid.activity_diff = abs( df_valid.pChEMBL_Value - df_valid.activity_calc )

        rows = df_valid.loc[ df_valid.activity_diff > crit, : ]
        cnt = rows.shape[0]
        if( 0 < cnt ) :
                print( '    moving ', cnt, 'samples ... : ', rows.diff )
                df_train = df_train.append( rows, ignore_index=True )
                df_valid.drop( rows.index, inplace=True )

                rowpred = df_train.sample( n=cnt )
                df_valid = df_valid.append( rowpred, ignore_index=True )
                df_train.drop( rowpred.index, inplace=True )

        return df_train, df_valid


def plot_xy( model, X_train, Y_train, X_valid, Y_valid, fname ) :
        YT_prediction = model.predict(X_train).flatten()
        YV_prediction = model.predict(X_valid).flatten()

        plt.figure( figsize=(4, 4) )
        plt.scatter(Y_train, YT_prediction, color='black', s=2)
        plt.scatter(Y_valid, YV_prediction, color='red', s=3)
        plt.xticks( np.arange(3, 12) )
        plt.yticks( np.arange(3, 12) )
        # plt.show()
        # plt.figure( figsize=(4, 4) )
        plt.savefig( fname, dpi=300 )
        plt.close('all')


def run_regression_iteration( sdfname, frac_test, nBits, fname_model ) :
        molesdf = readMolecules( sdfname )
        df_train, df_valid = setupFeatureDataFrame( molesdf, frac_test, nBits )

        for iter in range(99) :
                print( '\n--->ITER=', iter, ' : ', df_train.shape, df_valid.shape )
                X_train, Y_train = getFeature( df_train )
                X_valid, Y_valid = getFeature( df_valid )
                # model = do_regression( x_train, y_train, x_valid, y_valid )
                model, _ = make_regression_model( X_train, Y_train, X_valid, Y_valid, epochs=200, batch=10 )

                fname = "./PPARg-out/%s.%02d" % ( fname_model, iter )
                model.save( fname )
                fname += ".png"
                plot_xy( model, X_train, Y_train, X_valid, Y_valid, fname )

                df_train, df_valid = adjustTrainTestSet( df_train, df_valid, model, X_valid, Y_valid, crit=1.0 )







def main() :
        nBits = 2048
        frac_test = 0.3
        criteria = 6.0
        sdfname = "./200609-ChEMBL-2672.sdf"
        fname_model = './2001609-ChEMBL-2672-model.save'

        if False :              # classification
                sdfname = "./200303-chembl4qsar_6101.sdf"
                run_classification( sdfname, frac_test, nBits, criteria )
        elif False :            # regression-training
                run_regression( sdfname, frac_test, nBits, fname_model )
        elif True :
                run_regression_iteration( sdfname, frac_test, nBits, fname_model )
        else :                  # regression-prediction
                model = models.load_model( fname_model )
                do_prediction( model, '200610-biocides.sdf' )



if __name__ == '__main__' :
        main()
