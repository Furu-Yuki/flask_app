# 必要なモジュールのインポート
from flask import Flask, render_template, request, redirect
import io
import base64
import cv2
import numpy as np

# Flask をインスタンス化
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

# 拡張子が適切かどうかをチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods=['GET', 'POST'])
def predicts():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allowed_file(file.filename):

            # 画像処理を実行
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)  # OpenCV形式に変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # グレースケール変換
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # Cannyエッジ検出

            # 輪郭を検出
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 長方形を検出
            rectangles = []
            for contour in contours:
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    rectangles.append(approx)

            # 長方形を描画
            for rect in rectangles:
                cv2.drawContours(image, [rect], -1, (0, 0, 255), 2)  # 赤い枠を描画

            # 画像をバイナリデータに変換し、base64にエンコード
            _, buffer = cv2.imencode('.png', image)
            base64_str = base64.b64encode(buffer).decode('utf-8')
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # 描画した画像を保存
            cv2.imwrite('static/result_image.png', image)

            message_ = '枠線が認識されました。'
            return render_template('result.html', message=message_, image=base64_data)
        return redirect(request.url)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')

# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)
