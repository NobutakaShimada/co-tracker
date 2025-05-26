import cv2
import time

def test_device(device_path, description, display_time=3):
    """指定されたデバイスパスでカメラをテストする"""
    print(f"\nテスト: {description} ({device_path})")
    
    # デバイスを開く
    cap = cv2.VideoCapture(device_path)
    
    if not cap.isOpened():
        print(f"エラー: {device_path} を開けませんでした")
        return False
    
    # 解像度と設定を表示
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"カメラ設定: {width}x{height} @ {fps}fps")
    
    # フレームを取得
    ret, frame = cap.read()
    
    if not ret:
        print("エラー: フレームを取得できませんでした")
        cap.release()
        return False
    
    print(f"フレーム取得成功: {frame.shape[1]}x{frame.shape[0]}")
    
    # ウィンドウを作成して表示
    window_name = f"Test: {description}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 指定された時間だけフレームを表示（リアルタイム更新）
    start_time = time.time()
    frames_captured = 0
    
    while time.time() - start_time < display_time:
        ret, frame = cap.read()
        if ret:
            frames_captured += 1
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    elapsed_time = time.time() - start_time
    actual_fps = frames_captured / elapsed_time if elapsed_time > 0 else 0
    print(f"実際のFPS: {actual_fps:.2f} ({frames_captured}フレーム / {elapsed_time:.2f}秒)")
    
    # テスト画像を保存
    if frames_captured > 0:
        filename = f"c920_{description.replace(' ', '_')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"テスト画像を保存しました: {filename}")
    
    # 後片付け
    cap.release()
    cv2.destroyWindow(window_name)
    return True

# 異なるバックエンドとデバイスの組み合わせをテスト
def test_all_combinations():
    devices_to_test = [
        (0, "インデックス 0"),
        (1, "インデックス 1"),
        ("/dev/video0", "パス /dev/video0"),
        ("/dev/video1", "パス /dev/video1"),
        ("v4l2:///dev/video0", "V4L2 パス /dev/video0"),
        ("v4l2:///dev/video1", "V4L2 パス /dev/video1")
    ]
    
    successful_tests = []
    
    for device, description in devices_to_test:
        if test_device(device, description):
            successful_tests.append((device, description))
    
    # 結果をまとめる
    print("\n=== テスト結果まとめ ===")
    if successful_tests:
        print("成功した組み合わせ:")
        for device, description in successful_tests:
            print(f"- {description} ({device})")
        
        # 最適な組み合わせを推奨
        best_device, best_desc = successful_tests[0]
        print(f"\n推奨設定: {best_desc} ({best_device})")
    else:
        print("すべてのテストが失敗しました")

if __name__ == "__main__":
    print("C920 ウェブカメラテスト")
    print("OpenCVバージョン:", cv2.__version__)
    print("終了するには、画像ウィンドウを選択して 'q' キーを押してください")
    
    test_all_combinations()

