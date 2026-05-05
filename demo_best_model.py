#!/usr/bin/env python3
"""
Demo: En iyi LightGBM modelini kullanarak örnek bir session için top-20 tavsiye.
"""

import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from src.otto_phase1.config import BaselineConfig
from src.otto_phase1.popularity import build_popularity_from_train
from src.otto_phase1.covisitation import build_covisitation_from_train
from src.otto_phase1.io_utils import load_train_shards, load_test_shards
from src.otto_phase1.candidates import build_candidate_pool
from src.otto_phase1.scoring import compute_candidate_feature_and_score_maps


def load_lgb_model(model_path: Path):
    """LightGBM modelini yükle (native format)."""
    return lgb.Booster(model_file=str(model_path))


def run_demo():
    print("=" * 70)
    print("OTTO Phase 1 - En İyi Model Demo (LightGBM v2)")
    print("=" * 70)
    
    # Config
    config = BaselineConfig()
    data_root = Path("archive")
    models_dir = Path("outputs/lgb_models_full_clean_v2_full_targetwise")
    
    # Popularity ve Co-visitation üret (train'den)
    print("\n[1/5] Train verisi yükleniyor...")
    train_shards = load_train_shards(data_root)
    
    print("[2/5] Popularity hesaplanıyor...")
    popularity = build_popularity_from_train(train_shards)
    
    print("[3/5] Co-visitation grafiği oluşturuluyor...")
    train_shards = load_train_shards(data_root)
    covisitation = build_covisitation_from_train(train_shards, config)
    
    # LightGBM modellerini yükle
    print("[4/5] LightGBM modelleri yükleniyor...")
    lgb_models = {}
    for target in ["clicks", "carts", "orders"]:
        model_path = models_dir / f"lgb_model_{target}.txt"
        lgb_models[target] = load_lgb_model(model_path)
        print(f"  ✓ {target} modeli yüklendi")
    
    # Test veri'sinden örnek session al
    print("[5/5] Test verisi yükleniyor...")
    test_shards = load_test_shards(data_root)
    
    example_sessions = []
    for shard_df in test_shards:
        for session_id, session_df in shard_df.groupby("session", sort=False):
            events = list(zip(
                session_df["aid"].tolist(),
                session_df["ts"].tolist(),
                session_df["type"].astype("string").tolist(),
            ))
            if len(events) >= 3:  # En az 3 event olsun
                example_sessions.append((session_id, events))
                if len(example_sessions) >= 5:
                    break
        if len(example_sessions) >= 5:
            break
    
    # Demo: Her örnek session için tahmin yap
    print("\n" + "=" * 70)
    print("DEMO: TOP-20 ÖNERİ")
    print("=" * 70)
    
    for demo_idx, (session_id, session_events) in enumerate(example_sessions[:3], 1):
        print(f"\n📱 Session #{demo_idx} (ID: {session_id})")
        print(f"   Etkileşim sayısı: {len(session_events)}")
        print(f"   Son 3 ürün: {[e[0] for e in session_events[-3:]]}")
        
        # Aday havuzu oluştur
        candidate_pool = build_candidate_pool(
            session_events=session_events,
            popularity=popularity,
            covisitation=covisitation,
            config=config,
        )
        
        print(f"\n   Aday sayısı: {len(candidate_pool.aids)}")
        
        # Her target için tahmin yap
        for target in ["clicks", "carts", "orders"]:
            feature_map, score_map = compute_candidate_feature_and_score_maps(
                session_events=session_events,
                candidate_pool=candidate_pool,
                popularity=popularity,
                config=config,
                target=target,
            )
            
            # LightGBM öngörüsü
            X = np.array([
                [
                    feature_map.get(aid, {}).get(name, 0.0)
                    for name in [
                        "hist_click_score", "hist_cart_score", "hist_order_score",
                        "hist_presence", "covis_signal", "is_last_item",
                        "pop_target_inv_rank", "pop_weighted_inv_rank", "pop_global_inv_rank",
                        "target_weight_hist_click", "target_weight_hist_cart", "target_weight_hist_order",
                        "target_weight_hist_presence", "target_weight_covis", "target_weight_popular",
                        "target_weight_last_item"
                    ]
                ]
                for aid in candidate_pool.aids
            ], dtype=np.float32)
            
            lgb_proba = lgb_models[target].predict(X)
            
            # Top-20 seç
            top_20_idx = np.argsort(lgb_proba)[-20:][::-1]
            top_20_aids = [candidate_pool.aids[i] for i in top_20_idx]
            top_20_scores = [lgb_proba[i] for i in top_20_idx]
            
            print(f"\n   🎯 {target.upper()} - TOP-20 ÖNERİ:")
            for rank, (aid, score) in enumerate(zip(top_20_aids[:10], top_20_scores[:10]), 1):
                heur_score = score_map.get(aid, 0.0)
                print(f"      {rank:2d}. AID={aid:7d} | Model: {score:.4f} | Heuristic: {heur_score:.4f}")
            if len(top_20_aids) > 10:
                print(f"      ... (+ {len(top_20_aids) - 10} daha)")


if __name__ == "__main__":
    run_demo()
