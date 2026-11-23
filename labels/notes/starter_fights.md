# Starter fights: notes

Notes on the 15 starter fights in `labels/fights.csv` (section 0.2). All times are in the original downloaded video, as VLC shows them. Never trim or re-encode the files, or these times stop matching.

The same fouls are in `labels/notes/known_fouls.csv` in seconds, for code to read (the stoppage finder check in 2.2, golden set picks).

## Known fouls

Ari's video notes. Format: window in video, then foul timestamps (blue to red means the blue fighter fouled the red fighter).

- ufc321_aspinall_gane (qytyQY5Tf0s): window 04:13-13:00. Eye poke 08:43 blue to red, ref paused, clear live. Replays from 08:55.
- ufc285_jones_gane (M4DE1TRJvns): window 0:43-03:16. Low blow 0:47 blue to red, ref paused, clear live. Replays from 01:01.
- ufcfn20251101_cortesacosta_delija (KHIcHq0ZGPI): window 0:07-09:17. Eye poke 03:29 blue to red, fight stopped then restarted after evaluation, hard to see live. Clear replay 04:40.
- ufc326_garbrandt_long (jz-MtNqMTX0): window 0:40-22:58. Low blows 13:30, 14:40, 17:56 blue to red, ref paused, clear live. Replays from 13:48, 15:13, 18:27. Low quality video but fine.
- ufcfn20251018_holland_malott (wTbKaIYz2vE): window 01:46-26:11. Low blows 05:02 and 05:52 blue to red, ref paused, clear live. Replays from 05:16 and 6:24.
- ufcfn20240316_battle_loosa (agIWIQAb6Ns): window 01:59-09:56. Eye poke 08:09 red to blue, ref paused, clear live. Replay from 08:26.
- ufc276_munhoz_omalley (e6DTwSONguQ): window 02:19-14:08. Eye poke 12:07 blue to red, ref paused, clear live. Replay from 12:29.
- ufcfn20241214_kape_silva (haL_EjK7rVo): window 0:10-15:28. Low blows 02:45 and 07:54 blue to red, ref paused, clear live, replays from 03:02 and 08:18. Eye poke 13:33 blue to red, no pause, clear but quick, no replay.
- ufc300_gaethje_holloway (DdgEbBpPD_4): window 0:19-27:02. Eye poke 07:10 blue to red, ref paused, clearer in replay (replay starts 07:27, clear at 07:30). Eye poke 09:38 blue to red, ref paused, not clear live but very clear in replay (replay starts 09:55, clear at 9:58).
- ufcfn20210313_edwards_muhammad (Vzkfrw6GKfA): window 02:32-10:32. Eye poke 09:06 red to blue, ref paused, not clear live but clear in replay (replay starts 09:24, clear at 09:28). Low quality video. Also a low blow 02:56 blue to red, ref paused (found while labeling the mined queue in 2.3).
- ufc308_magomedov_petrosyan (m8Lngrf6ijQ): window 0:10-11:30. Listed as clean at 0.2, but labeling the mined queue in 2.3 found 4 low blows: 06:17 blue to red (offender being confirmed), 07:57 blue to red (contact unsure), 09:15 red to blue, 10:04 blue to red. Ref paused for all four.
- Clean (no fouls, no issues): Garry-Prates yruaaEGyxdg 0:44-26:38. Jandiroba-Dern DKmdkjrZT0g 0:40-26:07. Barber-Grasso aDXmHMQE1-A 2:00-17:21. O'Malley-Vera OY_ISlQoCvI 0:33-26:00.

## Corners (UFC.com event pages, red | blue)

Aspinall | Gane. Jones | Gane. Cortes Acosta | Delija. Garbrandt | Xiao Long. Holland | Malott. Battle | Loosa. Munhoz | O'Malley. Kape | Bruno Silva. Garry | Prates. Magomedov | Petrosyan. Jandiroba | Dern. Barber | Grasso. O'Malley | Vera. Gaethje | Holloway. Edwards | Muhammad.

## Confirming sources

- Gaethje vs Holloway: https://cagesidepress.com/2024/04/14/max-holloway-kos-justin-gaethje-in-final-second-of-ufc-300s-bmf-fight-in-electrifying-moment and https://www.si.com/fannation/wrestling/ufc-300
- Edwards vs Muhammad: https://www.fightersonlymag.com/latest-news/ufc-fight-night-leon-edwards-vs-belal-muhammad-recap-and-results/
- Battle vs Loosa: https://inkl.com/news/i-ll-f-cking-kill-you-ufc-fight-night-239-co-main-event-ends-in-near-brawl-after-no-contest
- Kape vs Silva: https://cagesidepress.com/2024/12/15/manel-kape-tkos-silva-f-bombs-everybody-including-adesanya
- Munhoz vs O'Malley: https://sports.yahoo.com/ufc-276-results-sean-o-023035406.html

## Dropped fights and why

Don't suggest these again unless the reason no longer applies.

| Fight | Why dropped |
|---|---|
| Cejudo vs Song (UFC Fight Night Seattle, 2025-02-22) | No verified full-fight video (only an unchecked 17:17 unofficial upload, id 2FbzornYnwU). It has both a low blow and an eye poke by Song in R3. Sources: https://www.cbssports.com/mma/news/ufc-fight-night-results-highlights-song-yadong-beats-henry-cejudo-after-eye-poke-forces-technical-decision , https://www.si.com/fannation/mma/news/bizarre-sequence-henry-cejudo-vs-song-yadong-ufc-seattle-awkward-ending , https://thescore.com/mma/news/3224448 |
| Rodriguez vs Xiaonan (UFC 272) | No full fight on YouTube |
| Marcos vs Aoriqileng | No video found |
| Thicknesse vs Musasa | No video found |
| Almeida vs Potieria | No video found |
| Diamond vs Radtke | No video found |
| Herbert vs Klein | No video found |
| Medic vs Holland | Too recent (no upload yet) |
| Silva vs Namajunas | Not in Ari's found list |
| Gane vs Pereira | Fouls weren't eye pokes or low blows |
