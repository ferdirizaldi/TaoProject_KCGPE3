# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:01:26 2022

@author: MSI
"""


UPLOAD_FOLDER ='static/uploads'

ALLOWED_EXTENSIONS = {'png','jpg','jpeg','gif'}

ML_MODEL_FILENAME = 'saved_model.h5'

import os

from flask import render_template
from flask import Flask, flash, request,redirect,url_for
from werkzeug.utils import secure_filename

import numpy as np

from tensorflow.keras.preprocessing import image
from keras.models import load_model
from keras.backend import set_session
import tensorflow as tf

app = Flask(__name__)

def load_model_from_file():
    mySession = tf.Session()
    set_session(mySession)
    myModel = load_model(ML_MODEL_FILENAME)
    myGraph = tf.get_default_graph()
    return (mySession,myModel,myGraph)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/guide',methods=['GET','POST'])
def guidePage():    
    if request.method == 'GET':
            return render_template('guide.html')

    
@app.route('/',methods=['GET','POST'])
def upload_file():
    if request.method =='GET':
        image_src = "/"+UPLOAD_FOLDER+"/resultSpace.jpg"
        pltX = []
        resultjl= [["-","-","-","-","-"]]
        return render_template('index.html',image_src=image_src,pltX=pltX,resultjl=resultjl)
    else:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename =='':
            flash('No Selected Files')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('unsupported')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            return redirect(url_for('uploaded_file',filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    test_image = image.load_img(UPLOAD_FOLDER+"/"+filename,target_size=(150,150))
    test_image = image.img_to_array(test_image)/255.
    test_image = np.expand_dims(test_image, axis=0)
    
    
    mySession = app.config['SESSION']
    myModel = app.config['MODEL'] 
    myGraph = app.config['GRAPH']
    
    
    with myGraph.as_default():
        set_session(mySession)
        result = myModel.predict(test_image)*100
        luna = myModel.predict_proba(test_image)*100
        print(luna)
        print(result)
        resultSplit = result.tolist()
        print (resultSplit)
        image_src = "/"+UPLOAD_FOLDER+"/"+filename
        # if result[0]<0.5:
        #     answer = "<div class='col text-center'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>guess:"+X+" "+str(result[0])+"</h4></div><div class='col'></div><div class='w-100'></div>"
        # else:
        #     answer = "<div class='col'></div><div class='col text-center'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>guess:"+Y+" "+str(result[0])+"</h4></div><div class='w-100'></div>"
        # results.append(answer)
        pltX ,pltY,resultFinal,resultjl = preparePlotChart(resultSplit[0])
        
        print(pltX)
        print(pltY)
        return render_template('index.html',scroll='resultBox',myResult=resultSplit[0],pltX=pltX,pltY=pltY,resultFinal=resultFinal,resultjl=resultjl,image_src=image_src)
    
def preparePlotChart(resultSplit):
    labelx = ['Black Grass/スズメノテッポウ(雀鉄砲）', 'Charlock/ノハラガラシ', 'Cleavers/ヤエムグラ（八重葎）','Common Chickweed/コハコベ（小繁縷）', 'Common Wheat/パンコムギ（麺麭小麦）', 'Fat Hen/しろざ（白藜）',
    'Loose Silky-bent/セイヨウヌカボ', 'Maize/トウモロコシ', 'Scentless Mayweed/イヌカミツレ（犬加密列)','Shepherds Purse/ナズナ(薺)', 'Small-flowered Cranesbill/チゴフウロ', 'Sugar Beet/てんさい（甜菜）']
    
    resultjll = [['Alopecurus aequalis','スズメノテッポウ','越年草','イネ科','生育期間10～6月。種子で繁殖。畑地、水田裏作、桑園などの春の代表的な雑草の一つで、発生量も多く強害草である。また、耕起前の水田では一面に密生し、大きな群落をつくるが耕起後の水田にはえることは少ない。'],
                 ['Sinapis arvensis ','ノハラガラシ','広葉草本/草本','アブラナ科','ノハラガラシは日本では市街地の道端や空き地に生える帰化種です。ハチやハエ、小さなチョウが花の蜜や花粉にやってきます。よく似た種であるシロガラシは、葉が細長く、種子のさやが毛深いことで見分けられます。'],
                 ['Galium spurium var. echinospermon','ヤエムグラ','一年草、越年草','アカネ科','林ややぶのまわり、人家の周辺や道ばたなどに広く生育します。秋の終わりから春の初めに発芽し、春から初夏によく成長し、黄緑色の小さな花をつけますが、夏には枯れます。茎、葉、果実ともにかぎ状のとげがたくさんついており、特に果実は衣服にくっついて、不快感を与えます。秋に発芽し、越冬して翌春にかけて生育する一年生冬雑草(越年草)。',],
                 ['Stellaria media','コハコベ','越年草','ナデシコ科','ヨーロッパ原産で、世界中に帰化している。花期は3～9月、直径 6～7 mm の花をつける。萼片は５枚。花弁は白色で５枚だが、基部まで深く裂けるため10枚に見える。'],
                 ['Triticum aestivum','コムギ','単子葉植物 1～2年草','イネ科','世界中で多くの品種がつくられ栽培されているコムギだが、その大部分は自然種ではなく、育成されたものだ。現在最も生産されているコムギはパンコムギ。これも、栽培の方法などから春まきコムギと秋まきコムギとに大別され、たくさんの品種がある。'],
                 ['Chenopodium album var. album','シロザ','一年草','ヒユ科','道ばたや荒れ地、畑などでみられ、高さ0.6～1mになる一年草。茎はよく枝分かれして直立します。畑でみられるものには基部から枝分かれして、高さ20cmほどにしかならないものもあります。'],
                 ['Apera spica-venti','セイヨウヌカボ','帰化','イネ科','一般にセイヨウヌカボヤまたはウインドグラスとして知られています。それらはヨーロッパ、北アフリカ、およびアジアの一部に自生していますが、北アメリカと南アメリカの多くで導入され帰化されています。'],
                 ['Zea mays','トウモロコシ','一年草','イネ科','トウモロコシの起源は定かではないが、北米南部から南米にかけて分布する野生種が原種とされている。現在栽培されている品種は、通常よく食べるスイートコーンのほかに、飼料用、ポップコーン用などの用途のちがいや形状などにより8種類に分けられる。'],
                 ['Asteraceae, Asterales, Magnoliopsida, Magnoliophyta','イヌカミツレ','一年草～二年草','キク科','ヨーロッパ原産。全体はほぼ無毛。葉は2〜3回深裂してからさらに糸状に細裂し、やや多肉質になる。始めはロゼットを形成する。茎は中部以上でよく分岐して高さは1mほどになる。'],
                 ['Capsella bursa-pastoris','ナズナ','被子植物 双子葉類 離べん花 越年草','アブラナ科','ナズナの花の咲く季節は、3月～7月くらいまで。真夏の酷暑の頃には段々とその姿を消し始めます。'+
                  'ナズナの花の見頃の季節は4月～5月。原っぱや河原、少し広い空き地などで群生するように咲き誇っている姿を見かけることがあります。'],
                 ['Geranium pusillum','チゴフウロ','一年草','フウロソウ科','チゴフウロはヨーロッパや西アジアに分布する、ピンクや紫色の花を咲かせる一年草です。庭や荒れ地、岩地など様々な環境に適合し、越冬できるほどの生命力を持ちます。フィンランドでは、畑の土手や放牧地などに群生しており、馴染み深い植物として知られています。'],
                 ['Beta vulgaris ssp. vulgaris','てんさい','被子植物','ヒユ科','テンサイは、ヒユ科アカザ亜科フダンソウ属。ヒユ科というと聞き慣れないかもしれませんが、ホウレンソウの仲間で、テンサイも育つとホウレンソウのような葉っぱが生えます。生育には涼しい地域が向いているため、日本では北海道で栽培されています。']]
    
    resultDict = dict(zip(labelx, resultSplit))
    print(resultDict)
    print("--------------------------")
    #sortingDict
    
    resultSorted = sorted(resultDict.items(), key=lambda x: x[1], reverse=True)
    print(resultSorted)
    print("--------------------------")
    
    resulty = []
    resultx = []    
    for i in range(5):
        resultx.append(resultSorted[i][0])
        resulty.append(resultSorted[i][1])
         
        
    print(resulty)
    print(resultx)
    
    resultjl = []
    resultFinal = []
    for i in range(len(resultx)):
        if resulty[i]>5: #maximum value only
            resultFinal.append(str(resultx[i])+"、確率"+str(resulty[i])+"%")
     
    for i in range(len(resultSplit)):
        if resultSplit[i]>5: #maximum value only info
            resultjl.append(resultjll[i])
            
    return resultx,resulty,resultFinal,resultjl
    
def main():
    (mySession,myModel,myGraph) = load_model_from_file();
    
    app.config['SECRET_KEY'] = 'super_secret_key'
    
    app.config['SESSION'] = mySession
    app.config['MODEL'] = myModel
    app.config['GRAPH'] = myGraph
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024
    app.run()
    
# results = []

main()

