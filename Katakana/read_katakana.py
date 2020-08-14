import struct
from PIL import Image
import numpy as np

def read_record_ETL1G(f):
    s = f.read(2052)
    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
    iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
    iL = iF.convert('P')
    return r + (iL,)

def read_kana():
    katakana = np.zeros([51, 1411, 63, 64], dtype=np.uint8) # 51 characters, 1411 writers, img size = 63*64
    for i in range(7,14):
        filename = 'ETL1/ETL1C_{:02d}'.format(i)
        with open(filename, 'rb') as f:
            if i!=13: # file 13 only has 3 characters, others have 8 characters
                limit = 8
            else:
                limit=3
            for dataset in range(limit):
                for j in range(1411):
                    try :
                        r = read_record_ETL1G(f)
                        katakana[(i - 7) * 8 + dataset, j] = np.array(r[-1])
                    except struct.error: # two imgs are blank according to the ETL website, so this prevents any errors
                        pass
    np.savez_compressed("kana.npz", katakana)

read_kana()
