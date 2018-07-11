from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint

def callbacks(phase='mdl_best.h5'):
    cb_log = CSVLogger(filename='log.csv', append=True)
    cb_cpt_best = ModelCheckpoint(filepath=phase, monitor='val_acc', save_best_only=True, save_weights_only=True, verbose=1)
	#cb_cpt_last = ModelCheckpoint(filepath='mdl_last.h5', monitor='val_acc', save_best_only=False, save_weights_only=True, verbose=0)
    
    return [cb_log, cb_cpt_best]
