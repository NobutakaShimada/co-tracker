# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
import numpy as np
import cv2
import time
from collections import deque

from cotracker.predictor import CoTrackerOnlinePredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--camera_path",
        type=str,
        default=None,
        help="カメラデバイスパス (例: /dev/video0)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=20, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="640x480",
        help="カメラ解像度 (WxH, 例: 1280x720)",
    )
    parser.add_argument(
        "--display_scale",
        type=float,
        default=1.0,
        help="表示ウィンドウの拡大率",
    )
    parser.add_argument(
        "--max_points", 
        type=int, 
        default=3000,
        help="表示する最大ポイント数"
    )
    parser.add_argument(
        "--trail_length",
        type=int,
        default=10,
        help="表示する軌跡の長さ (デフォルト: 10フレーム)"
    )
    parser.add_argument(
        "--display_mode",
        choices=["points", "trails", "both"],
        default="both",
        help="表示モード (points: 点のみ, trails: 軌跡のみ, both: 両方)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="詳細な時間計測を表示"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="時間ウィンドウのサイズ (デフォルト: 16フレーム)"
    )
    parser.add_argument(
        "--inference_interval",
        type=int,
        default=8,
        help="推論実行間隔 (デフォルト: 8フレーム毎)"
    )

    args = parser.parse_args()

    # カメラを初期化
    if args.camera_path:
        print(f"デバイスパスを使用: {args.camera_path}")
        cap = cv2.VideoCapture(args.camera_path)
    else:
        print(f"カメラインデックスを使用: {args.camera_id}")
        cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        raise ValueError(f"カメラを開けませんでした。別のカメラパスやIDを試してください。")
    
    # 解像度設定
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"カメラ解像度を {width}x{height} に設定しました")
        except:
            print(f"解像度設定エラー: {args.resolution}")
    
    # 実際の設定を確認
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"実際のカメラ設定: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # モデルを読み込み
    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        print("モデルをロード中...")
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)
    print(f"モデルを {DEFAULT_DEVICE} に読み込みました")
    
    # パラメータ設定
    window_size = args.window_size
    inference_interval = args.inference_interval
    print(f"ウィンドウサイズ: {window_size}フレーム")
    print(f"推論間隔: {inference_interval}フレーム毎")

    window_frames = []
    pred_tracks = None
    pred_visibility = None
    is_first_step = True
    frame_count = 0
    start_time = time.time()
    fps_list = []  # FPS平均用
    add_new_grid = False  # 新しいグリッドを追加するフラグ
    
    # 時間計測用の変数（初期値として0.0を入れておく）
    inference_times = deque([0.0], maxlen=10)  # 推論時間
    drawing_times = deque([0.0], maxlen=10)    # 描画時間
    capture_times = deque([0.0], maxlen=10)    # キャプチャ時間
    total_times = deque([0.0], maxlen=10)      # 全体の処理時間
    
    # 表示モード (0: 点のみ, 1: 軌跡のみ, 2: 両方)
    display_mode_map = {"points": 0, "trails": 1, "both": 2}
    current_display_mode = display_mode_map[args.display_mode]
    
    # 過去のフレームを保存するバッファ
    frame_buffer = deque(maxlen=args.trail_length)
    
    # 時間計測表示フラグ
    show_timing = args.profile

    print(f"CoTracker: {DEFAULT_DEVICE}で実行中。")
    print("操作キー:")
    print("  'q': 終了")
    print("  'r': トラッキングをリセット")
    print("  'g': 現在のフレームに新しいグリッドを追加")
    print("  '+': グリッドサイズを増加")
    print("  '-': グリッドサイズを減少")
    print("  'm': 表示モード切替 (点のみ/軌跡のみ/両方)")
    print("  '[': 軌跡の長さを短くする")
    print("  ']': 軌跡の長さを長くする")
    print("  'p': 時間計測情報の表示/非表示")
    print("  'w': ウィンドウサイズを増減 (Shift+w: 減少)")
    print("  'i': 推論間隔を増減 (Shift+i: 減少)")

    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
        # フレームをモデル用の形式に変換
        frames_to_use = window_frames[-window_size:]
        if len(frames_to_use) < window_size:
            # ウィンドウサイズに満たない場合、足りない分を最初のフレームで埋める
            padding = [frames_to_use[0]] * (window_size - len(frames_to_use))
            frames_to_use = padding + frames_to_use
        
        video_chunk = (
            torch.tensor(
                np.stack(frames_to_use), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )

    # ウィンドウを作成
    cv2.namedWindow('CoTracker Live Demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CoTracker Live Demo', 800, 600)

    # 現在のグリッドサイズ
    current_grid_size = args.grid_size
    
    # 現在の軌跡の長さ
    current_trail_length = args.trail_length

    # 色テーブルの事前計算
    def generate_color_table(max_colors):
        color_table = []
        for i in range(max_colors):
            hue = int((i / max_colors * 180) % 180)
            color = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]))
            color_table.append(color)
        return color_table
    
    # 1000色の色テーブルを事前に用意
    color_table = generate_color_table(1000)

    while True:
        loop_start_time = time.time()
        
        # キャプチャ時間計測開始
        capture_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("カメラからのフレーム取得に失敗しました")
            break
        
        # キャプチャ時間計測終了
        capture_time = time.time() - capture_start
        capture_times.append(capture_time)

        # フレームバッファに最新フレームを追加
        frame_buffer.append(frame.copy())

        # 全体のFPSを計算
        if frame_count % 10 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            if elapsed > 0:
                current_fps = 10 / elapsed
                fps_list.append(current_fps)
                if len(fps_list) > 10:
                    fps_list.pop(0)
                avg_fps = sum(fps_list) / len(fps_list)
                #print(f"FPS: {avg_fps:.1f}, グリッドサイズ: {current_grid_size}x{current_grid_size}")
            start_time = current_time

        # BGRからRGBに変換（CoTrackerはRGBを期待）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        window_frames.append(frame_rgb)
        
        # 十分なフレームが集まったら処理
        if len(window_frames) >= window_size:
            inference_time = 0
            if frame_count % inference_interval == 0 and frame_count != 0 or add_new_grid:
                try:
                    # 推論時間計測開始
                    inference_start = time.time()
                    
                    if add_new_grid:
                        # 新しいグリッドを追加する場合
                        # grid_query_frameに現在のフレームインデックスを指定
                        current_grid_query_frame = len(window_frames) - 1
                        print(f"現在のフレームに新しいグリッド(サイズ:{current_grid_size})を追加...")
                        
                        # 既存の追跡点を保持するために、is_first_step=Falseを指定
                        pred_tracks, pred_visibility = _process_step(
                            window_frames,
                            is_first_step=False,  # 既存の追跡を維持
                            grid_size=current_grid_size,
                            grid_query_frame=current_grid_query_frame,  # 新しいフレームでグリッド生成
                        )
                        add_new_grid = False
                    else:
                        # 通常の追跡更新
                        pred_tracks, pred_visibility = _process_step(
                            window_frames,
                            is_first_step,
                            grid_size=current_grid_size,
                            grid_query_frame=args.grid_query_frame,
                        )
                    is_first_step = False
                    
                    # 推論時間計測終了
                    inference_time = time.time() - inference_start
                    inference_times.append(inference_time)
                    
                    if pred_tracks is not None:
                        #print(f"トラックを計算: {pred_tracks.shape}, 可視性: {pred_visibility.shape}")
                        pass
                except Exception as e:
                    print(f"トラック計算エラー: {e}")
                
                # スライディングウィンドウを維持
                if len(window_frames) > window_size * 2:  # バッファが大きくなりすぎないように制限
                    window_frames = window_frames[-window_size * 2:]
            
            # トラックが計算されていれば表示
            display_frame = frame.copy()
            
            # 描画時間計測開始
            drawing_start = time.time()
            
            if pred_tracks is not None and pred_visibility is not None:
                try:
                    # トラックデータをNumPy配列に変換
                    if isinstance(pred_tracks, torch.Tensor):
                        # 形状: [1, T, N, 2] -> [T, N, 2]
                        track_points = pred_tracks[0].detach().cpu().numpy()
                        if isinstance(pred_visibility, torch.Tensor):
                            # 形状: [1, T, N] -> [T, N]
                            visibility = pred_visibility[0].detach().cpu().numpy()
                        else:
                            visibility = pred_visibility
                    else:
                        track_points = pred_tracks
                        visibility = pred_visibility
                    
                    # 表示するポイント数を制限
                    max_display = min(track_points.shape[1], args.max_points)
                    
                    # ランダムに表示するポイントを選択
                    indices = np.random.choice(track_points.shape[1], max_display, replace=False)
                    
                    # 使用する軌跡の長さを決定（利用可能なフレーム数または設定値のいずれか小さい方）
                    trail_length = min(current_trail_length, track_points.shape[0])
                    
                    # 描画したポイントのカウント
                    points_drawn = 0
                    
                    # 各ポイントについて処理
                    for i, point_idx in enumerate(indices):
                        color = color_table[point_idx % len(color_table)]
                        
                        # 最新のフレームのポイントを表示（display_modeが0または2の場合）
                        if current_display_mode in [0, 2]:  # 点のみ、または両方
                            last_point = track_points[-1, point_idx]
                            last_vis = visibility[-1, point_idx]
                            
                            # 可視性をチェック
                            is_visible = True
                            if isinstance(last_vis, np.ndarray):
                                is_visible = last_vis.item() if last_vis.size == 1 else bool(last_vis.any())
                            else:
                                is_visible = bool(last_vis)
                            
                            if is_visible:
                                if isinstance(last_point, np.ndarray):
                                    if last_point.ndim > 0 and last_point.size >= 2:
                                        x = float(last_point[0])
                                        y = float(last_point[1])
                                    else:
                                        continue
                                else:
                                    x, y = float(last_point[0]), float(last_point[1])
                                
                                # 整数座標に変換
                                x, y = int(round(x)), int(round(y))
                                
                                # 画像範囲内かチェック
                                if 0 <= x < display_frame.shape[1] and 0 <= y < display_frame.shape[0]:
                                    # 点を描画
                                    cv2.circle(display_frame, (x, y), 3, color, -1)
                                    points_drawn += 1
                        
                        # 軌跡を表示（display_modeが1または2の場合）
                        if current_display_mode in [1, 2]:  # 軌跡のみ、または両方
                            # 利用可能なフレーム数を考慮
                            available_frames = min(trail_length, track_points.shape[0])
                            
                            # 過去のいくつかのフレームから軌跡を作成
                            trajectory_points = []
                            
                            for t in range(1, available_frames + 1):
                                frame_idx = -t  # 最新から過去に向かってt番目
                                point = track_points[frame_idx, point_idx]
                                vis = visibility[frame_idx, point_idx]
                                
                                # 可視性をチェック
                                is_visible = True
                                if isinstance(vis, np.ndarray):
                                    is_visible = vis.item() if vis.size == 1 else bool(vis.any())
                                else:
                                    is_visible = bool(vis)
                                
                                if is_visible:
                                    if isinstance(point, np.ndarray):
                                        if point.ndim > 0 and point.size >= 2:
                                            x = float(point[0])
                                            y = float(point[1])
                                        else:
                                            continue
                                    else:
                                        x, y = float(point[0]), float(point[1])
                                    
                                    # 整数座標に変換
                                    x, y = int(round(x)), int(round(y))
                                    
                                    # 画像範囲内かチェック
                                    if 0 <= x < display_frame.shape[1] and 0 <= y < display_frame.shape[0]:
                                        trajectory_points.append((x, y))
                            
                            # 軌跡を描画（連続した点を線で結ぶ）
                            for j in range(1, len(trajectory_points)):
                                # 過去に向かうほど薄い色にする
                                alpha = 0.8 * (j / len(trajectory_points))  # 透明度係数（古いほど薄く）
                                # 色を薄くする（背景色に近づける）
                                faded_color = tuple(int(c * alpha + 255 * (1-alpha)) for c in color)
                                
                                cv2.line(display_frame, trajectory_points[j-1], trajectory_points[j], faded_color, 1)
                    
                    # 描画したポイント数を表示
                    cv2.putText(
                        display_frame, 
                        f"Tracking {points_drawn} points", 
                        (10, display_frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        1
                    )
                except Exception as e:
                    print(f"描画エラー: {e}")
            
            # 描画時間計測終了
            drawing_time = time.time() - drawing_start
            drawing_times.append(drawing_time)
            
            # FPS表示
            if len(fps_list) > 0:
                avg_fps = sum(fps_list) / len(fps_list)
                cv2.putText(
                    display_frame, 
                    f"FPS: {avg_fps:.1f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
            
            # グリッドサイズ表示
            cv2.putText(
                display_frame, 
                f"Grid: {current_grid_size}x{current_grid_size}", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # ウィンドウサイズと推論間隔を表示
            cv2.putText(
                display_frame, 
                f"Window: {window_size}, Interval: {inference_interval}", 
                (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # 表示モードと軌跡の長さを表示
            mode_names = ["Points Only", "Trails Only", "Points & Trails"]
            cv2.putText(
                display_frame, 
                f"Mode: {mode_names[current_display_mode]}, Trail Length: {current_trail_length}", 
                (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # 時間計測情報の表示
            if show_timing:
                # 平均値を計算 (ゼロ除算エラーを防ぐ)
                avg_inference = sum(inference_times) / max(len(inference_times), 1)
                avg_drawing = sum(drawing_times) / max(len(drawing_times), 1)
                avg_capture = sum(capture_times) / max(len(capture_times), 1)
                
                # 時間計測情報を表示
                y_pos = 150
                cv2.putText(
                    display_frame, 
                    f"Inference: {avg_inference*1000:.1f}ms", 
                    (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 255), 
                    1
                )
                y_pos += 25
                cv2.putText(
                    display_frame, 
                    f"Drawing: {avg_drawing*1000:.1f}ms", 
                    (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 255), 
                    1
                )
                y_pos += 25
                cv2.putText(
                    display_frame, 
                    f"Capture: {avg_capture*1000:.1f}ms", 
                    (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 255), 
                    1
                )
            
            # バッファリング状況表示
            if pred_tracks is None:
                cv2.putText(
                    display_frame, 
                    f"Buffering: {len(window_frames)}/{window_size}", 
                    (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
            
            # ヘルプテキスト
            cv2.putText(
                display_frame, 
                "q:終了 r:リセット g:グリッド追加 +/-:グリッド w/W:ウィンドウサイズ i/I:推論間隔", 
                (10, display_frame.shape[0] - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (200, 200, 200), 
                1
            )
            
            # 結果を表示
            cv2.imshow('CoTracker Live Demo', display_frame)
        else:
            # 十分なフレームがたまるまでは元のフレームを表示
            cv2.putText(
                frame, 
                f"Buffering frames... {len(window_frames)}/{window_size}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            cv2.imshow('CoTracker Live Demo', frame)
        
        # 1フレームの全体処理時間を計測
        total_time = time.time() - loop_start_time
        total_times.append(total_time)
        
        # 詳細な計測情報をコンソールに出力（10フレームごと）
        if args.profile and frame_count % 10 == 0 and len(total_times) > 0:
            avg_total = sum(total_times) / max(len(total_times), 1)
            avg_inference = sum(inference_times) / max(len(inference_times), 1)
            avg_drawing = sum(drawing_times) / max(len(drawing_times), 1)
            avg_capture = sum(capture_times) / max(len(capture_times), 1)
            other_time = avg_total - (avg_inference + avg_drawing + avg_capture)
            
            print("\n===== 時間計測情報 =====")
            print(f"全体処理時間:     {avg_total*1000:.1f}ms (FPS: {1/avg_total:.1f})")
            print(f"キャプチャ時間:   {avg_capture*1000:.1f}ms ({avg_capture/avg_total*100:.1f}%)")
            print(f"推論時間:         {avg_inference*1000:.1f}ms ({avg_inference/avg_total*100:.1f}%)")
            print(f"描画時間:         {avg_drawing*1000:.1f}ms ({avg_drawing/avg_total*100:.1f}%)")
            print(f"その他処理時間:   {other_time*1000:.1f}ms ({other_time/avg_total*100:.1f}%)")
            print("========================\n")
        
        frame_count += 1
        
        # キー入力を処理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("終了中...")
            break
        elif key == ord('r'):
            print("トラッキングをリセット中...")
            window_frames = []
            pred_tracks = None
            pred_visibility = None
            is_first_step = True
            frame_buffer.clear()
            # 時間計測データをリセット (ゼロ除算を防ぐため0.0を入れておく)
            inference_times = deque([0.0], maxlen=10)
            drawing_times = deque([0.0], maxlen=10)
            capture_times = deque([0.0], maxlen=10)
            total_times = deque([0.0], maxlen=10)
        elif key == ord('g'):
            # 現在のフレームに新しいグリッドを追加
            if len(window_frames) >= window_size:
                add_new_grid = True
            else:
                print("バッファが不足しています。フレームがたまるまで待ってください。")
        elif key == ord('+') or key == ord('='):
            # グリッドサイズを増加
            current_grid_size = min(current_grid_size + 5, 50)
            print(f"グリッドサイズを {current_grid_size} に変更しました")
        elif key == ord('-') or key == ord('_'):
            # グリッドサイズを減少
            current_grid_size = max(current_grid_size - 5, 5)
            print(f"グリッドサイズを {current_grid_size} に変更しました")
        elif key == ord('m'):
            # 表示モードを切り替え (0: 点のみ, 1: 軌跡のみ, 2: 両方)
            current_display_mode = (current_display_mode + 1) % 3
            mode_names = ["Points Only", "Trails Only", "Points & Trails"]
            print(f"表示モードを '{mode_names[current_display_mode]}' に変更しました")
        elif key == ord('['):
            # 軌跡の長さを短くする
            current_trail_length = max(current_trail_length - 1, 2)
            print(f"軌跡の長さを {current_trail_length} に変更しました")
        elif key == ord(']'):
            # 軌跡の長さを長くする
            current_trail_length = min(current_trail_length + 1, window_size)
            print(f"軌跡の長さを {current_trail_length} に変更しました")
        elif key == ord('p'):
            # 時間計測情報の表示/非表示切り替え
            show_timing = not show_timing
            print(f"時間計測情報表示: {'オン' if show_timing else 'オフ'}")
        elif key == ord('w'):
            # ウィンドウサイズを変更
            if cv2.waitKey(100) & 0xFF == ord('W'):  # Shift+w (大文字W)
                # ウィンドウサイズを小さくする
                window_size = max(window_size - 2, 4)
            else:
                # ウィンドウサイズを大きくする
                window_size = min(window_size + 2, 32)
            print(f"ウィンドウサイズを {window_size} に変更しました")
            # リセットする必要がある
            is_first_step = True
        elif key == ord('i'):
            # 推論間隔を変更
            if cv2.waitKey(100) & 0xFF == ord('I'):  # Shift+i (大文字I)
                # 推論間隔を小さくする
                inference_interval = max(inference_interval - 1, 1)
            else:
                # 推論間隔を大きくする
                inference_interval = min(inference_interval + 1, 16)
            print(f"推論間隔を {inference_interval} に変更しました")
            
    # リソース解放
    cap.release()
    cv2.destroyAllWindows()
    print("デモを終了しました。")
