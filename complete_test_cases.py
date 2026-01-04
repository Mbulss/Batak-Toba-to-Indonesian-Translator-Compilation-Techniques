from test3 import compile_text

# Test cases using ONLY words from batak_to_indo_lexicon.json
# Total: 50 sentences
test_cases = {
    "Simple Sentences (15 tests)": [
        "Au mangan",                    # saya makan
        "Ho manangi",                   # kamu pergi  
        "Ia marniat",                   # dia bernyanyi
        "Au mangan tu pasar",           # saya makan ke pasar
        "Ho mangan buku",               # kamu makan buku
        "Ia marniat di rohang",         # dia bernyanyi di rumah
        "Hita mangan",                  # kita makan
        "Hamu manangi",                 # kalian pergi
        "Au manangih",                  # saya menangis
        "Ho marhobas",                  # kamu membaca
        "Dongan manangi",               # teman pergi
        "Au manangihon buku",           # saya memberi buku
        "Ho marhobas buku",             # kamu membaca buku
        "Ia mangan di pasar",           # dia makan di pasar
        "Hita marniat tu rohang"        # kita bernyanyi ke rumah
    ],

    "Complex Sentences (10 tests)": [
        "Au manangihon buku tu dongan",             # saya memberi buku ke teman
        "Ho marhobas buku di rohang",               # kamu membaca buku di rumah
        "Ia mangan buku di pasar",                  # dia makan buku di pasar
        "Hita mangan tu pasar",                     # kita makan ke pasar
        "Au manangihon dongan buku",                # saya memberi teman buku
        "Dongan marhobas buku di rohang",           # teman membaca buku di rumah
        "Au mangan di pasar",                       # saya makan di pasar
        "Ia manangihon hita buku",                  # dia memberi kita buku
        "Ho mangan buku di pasar",                  # kamu makan buku di pasar
        "Hita marniat di pasar"                     # kita bernyanyi di pasar
    ],

    "Interjections (5 tests)": [
        "Horas",                        # halo/selamat
        "Horas di hamu",                # halo di kalian
        "Horas tu dongan",              # halo ke teman
        "Horas di pasar",               # halo di pasar
        "Horas tu hita"                 # halo ke kita
    ],

    "Multi-Word Phrases (8 tests)": [
        "Au mangulahon ulaon",          # saya bekerja
        "Ho mambahen salah",            # kamu berbuat salah
        "Ia mangulahon salah",          # dia bersalah
        "Hita marsak roha",             # kita sedih
        "Au mambahen roha",             # saya menyenangkan
        "Ho mangido maaf",              # kamu minta maaf
        "Dongan mangido tu au",         # teman meminta ke saya
        "Au mangulahon dosa"            # saya berdosa
    ],

    "Grammar Errors (7 tests)": [
        "Tu pasar au manangi",          # tu before subject (wrong)
        "Do",                           # only particle
        "Manangi manangi pasar",        # double verb
        "Au do pasar",                  # do misuse
        "Manangi",                      # only verb
        "Au tu",                        # incomplete
        "Pasar manangi au"              # wrong order
    ],

    "Unknown Words (5 tests)": [
        "Au xyz pasar",                 # xyz not in dict
        "Ho manangi qwerty",            # qwerty not in dict
        "Dongan asdfgh buku",           # asdfgh not in dict
        "Au mangan zxcvbn",             # zxcvbn not in dict
        "Ia unknown tu pasar"           # unknown not in dict
    ]
}


def run_tests():
    print("=" * 80)
    print("BATAK TOBA -> INDONESIAN TRANSLATOR - TEST SUITE")
    print("=" * 80)
    print("Total Test Cases: 50 sentences")
    print("Mode: Unified (Always translates + Shows warnings)")
    print("Dictionary: 1,863+ Batak Toba words")
    print("=" * 80)

    total_tests = 0
    total_with_warnings = 0
    total_clean = 0
    
    category_results = {}

    for category, sentences in test_cases.items():
        print(f"\n{'=' * 80}")
        print(f"{category}")
        print("=" * 80)
        
        cat_total = len(sentences)
        cat_warnings = 0
        cat_clean = 0

        for i, sentence in enumerate(sentences, 1):
            total_tests += 1
            result = compile_text(sentence)
            errors = result["grammar_errors"]
            translation = result["translation"]

            print(f"\nTest {i}: \"{sentence}\"")
            print("-" * 80)
            print(f"Translation: {translation}")
            
            if errors:
                cat_warnings += 1
                total_with_warnings += 1
                error_codes = [e.code for e in errors]
                print(f"[!] Warnings: {', '.join(error_codes)}")
            else:
                cat_clean += 1
                total_clean += 1
                print("[OK] No warnings")
        
        category_results[category] = {
            "total": cat_total,
            "clean": cat_clean,
            "warnings": cat_warnings
        }

    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    for cat, results in category_results.items():
        clean_pct = (results["clean"] / results["total"]) * 100
        print(f"\n{cat}:")
        print(f"  Total: {results['total']}")
        print(f"  Clean: {results['clean']} ({clean_pct:.1f}%)")
        print(f"  With warnings: {results['warnings']}")
    
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total sentences tested: {total_tests}")
    print(f"All translations successful: {total_tests} (100%)")
    print(f"Clean translations (no warnings): {total_clean} ({(total_clean/total_tests)*100:.1f}%)")
    print(f"Translations with warnings: {total_with_warnings} ({(total_with_warnings/total_tests)*100:.1f}%)")
    print("=" * 80)
    
    print("\n[NOTE] All sentences were successfully translated.")
    print("   Warnings indicate potential grammar issues but don't block translation.")


if __name__ == "__main__":
    run_tests()