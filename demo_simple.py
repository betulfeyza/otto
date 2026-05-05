#!/usr/bin/env python3
"""
Demo: En iyi LightGBM modelini kullanarak örnek session için top-20 tavsiye.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.otto_phase1.config import BaselineConfig
from src.otto_phase1.popularity import build_popularity_from_train
from src.otto_phase1.covisitation import build_covisitation_from_train
from src.otto_phase1.io_utils import iter_shard_dataframes
from src.otto_phase1.candidates import build_candidate_pool
from src.otto_phase1.scoring import compute_candidate_feature_and_score_maps


def main():
    print("=" * 80)
    print("🎬 OTTO Phase 1 - En İyi Model Demo (LightGBM v2 Full)")
    print("=" * 80)
    
    config = BaselineConfig()
    data_root = Path("archive")
    models_dir = Path("outputs/lgb_models_full_clean_v2_full_targetwise")
    
    # Step 1: Train verisi ve modeller
    print("\n📊 [1/4] Train verisi yükleniyor...")
    train_shards = list(iter_shard_dataframes(data_root / "train_parquet"))
    print(f"   ✓ {len(train_shards)} shard yüklendi")
    
    print("📊 [2/4] Popularity ve co-visitation hesaplanıyor...")
    popularity = build_popularity_from_train(train_shards, config)
    covisitation = build_covisitation_from_train(train_shards, config)
    print(f"   ✓ Popularity ve co-visitation ready")
    
    print("🤖 [3/4] LightGBM modelleri yükleniyor...")
    lgb_models = {}
    for target in ["clicks", "carts", "orders"]:
        model_path = models_dir / f"lgb_model_{target}.txt"
        lgb_models[target] = lgb.Booster(model_file=str(model_path))
    print(f"   ✓ 3 model yüklendi (clicks, carts, orders)")
    
    # Step 2: Örnek sessions seç
    print("\n🔍 [4/4] Train'den örnek sessions seçiliyor...")
    example_sessions = []
    for shard_df in train_shards:
        for session_id, session_df in shard_df.groupby("session", sort=False):
            events = list(zip(
                session_df["aid"].astype(int).tolist(),
                session_df["ts"].astype(int).tolist(),
                session_df["type"].astype(str).tolist(),
            ))
            if len(events) >= 5:
                example_sessions.append((session_id, events))
                if len(example_sessions) >= 3:
                    break
        if len(example_sessions) >= 3:
            break
    
    print(f"   ✓ {len(example_sessions)} örnek session seçildi\n")
    
    # Step 3: Demo predictions
    print("=" * 80)
    print("🎯 TOP-20 ÖNERİ DEMOSu")
    print("=" * 80)
    
    feature_names = [
        "hist_click_score", "hist_cart_score", "hist_order_score",
        "hist_presence", "covis_signal", "is_last_item",
        "pop_target_inv_rank", "pop_weighted_inv_rank", "pop_global_inv_rank",
        "target_weight_hist_click", "target_weight_hist_cart", "target_weight_hist_order",
        "target_weight_hist_presence", "target_weight_covis", "target_weight_popular",
        "target_weight_last_item"
    ]
    
    for demo_idx, (session_id, session_events) in enumerate(example_sessions, 1):
        print(f"\n📱 SESSION #{demo_idx} (ID: {int(session_id)})")
        print(f"   Toplam etkileşim: {len(session_events)}")
        print(f"   Son 3 ürün: {[int(e[0]) for e in session_events[-3:]]}")
        print(f"   Etkileşim tipleri: {set(e[2] for e in session_events)}")
        
        # Candidate pool
        candidate_pool = build_candidate_pool(
            session_events=session_events,
            popularity=popularity,
            covisitation=covisitation,
            config=config,
        )
        print(f"   Aday havuzu: {len(candidate_pool.aids)} ürün")
        
        # Her target için prediction
        for target in ["clicks", "carts", "orders"]:
            print(f"\n   🛍️  [{target.upper()}]")
            
            feature_map, score_map = compute_candidate_feature_and_score_maps(
                session_events=session_events,
                candidate_pool=candidate_pool,
                popularity=popularity,
                config=config,
                target=target,
            )
            
            # Features prepare
            X = np.array([
                [feature_map.get(aid, {}).get(fname, 0.0) for fname in feature_names]
                for aid in candidate_pool.aids
            ], dtype=np.float32)
            
            # LightGBM prediction
            lgb_scores = lgb_models[target].predict(X)
            
            # Top-20
            top_indices = np.argsort(lgb_scores)[-20:][::-1]
            
            print(f"      Rank │    AID    │ LGB Score │ Heuristic")
            print(f"      ─────┼───────────┼───────────┼──────────")
            for rank, idx in enumerate(top_indices[:10], 1):
                aid = int(candidate_pool.aids[idx])
                lgb_score = float(lgb_scores[idx])
                heur_score = float(score_map.get(aid, 0.0))
                print(f"       {rank:2d}  │ {aid:7d}   │  {lgb_score:.4f}  │  {heur_score:.4f}")
            
            if len(top_indices) > 10:
                print(f"      ... ({len(top_indices) - 10} daha)")
    
    print("\n" + "=" * 80)
    print("✅ Demo tamamlandı!")
    print("=" * 80)


if __name__ == "__main__":
    main()
