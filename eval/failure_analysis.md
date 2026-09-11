# Failure analysis extract

## Track A — reply quality (Groq path)

Worst **15** by groundedness from `data/processed/reply_eval_with_retrieval.jsonl` (n=40).

### A1. id=921054 · groundedness=3 · judge_mean=3.67 · top_sim=0.20

- **Message:** iTunes wants me to sign in 3 times and says my "session has expired" is there any solution to this.. I've tried a re-install..
- **Reply:** It sounds like the sign‑in process is getting stuck. First, make sure your Mac’s date & time are set correctly and that you’re connected to a reliable network. Then sign out of iTunes, close the app, and reopen it to sign back in. If the problem persists, try signing in from anot
- **Top retrieved:** sim=0.20 | cust=`trying to launch Radio on Watch, it says to sign into iTunes in Watch app on phone. No such sign in exists. Help?` | agent=`We'd like to help. In DM, please let us know the software version being used on the paired iPhone.`
- **Heuristic why:** Token overlap with retrieved brand replies = 0; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A2. id=2515969 · groundedness=3 · judge_mean=3.67 · top_sim=0.23

- **Message:** hello! I’m in Louisville and cannot make any outbound calls and no one can call me. Help,please
- **Reply:** I’m sorry you’re having trouble making or receiving calls.  
First, check that cellular service is active and data is enabled on your phone.  
If the problem continues, let me know which carrier you’re on so we can dig deeper.
- **Top retrieved:** sim=0.23 | cust=`hello?` | agent=`We'd love to help. DM us the country where you are located. We can go from there.`
- **Heuristic why:** Token overlap with retrieved brand replies = 1; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A3. id=2479016 · groundedness=3 · judge_mean=3.67 · top_sim=0.21

- **Message:** do I need to make an appointment to see if I need a battery replacement?
- **Reply:** If your battery health is below 80 % or the device is noticeably slowing, a replacement may be needed.  
You can schedule a repair or bring the device in at an Apple Store or an authorized service provider.  
If you’d like, I
- **Top retrieved:** sim=0.21 | cust=`I️ need an update to make these boxes go away` | agent=`Let's work this out together. Meet us in DM so we can better assist you with this issue.`
- **Heuristic why:** Token overlap with retrieved brand replies = 1; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A4. id=1339024 · groundedness=3 · judge_mean=3.67 · top_sim=0.26

- **Message:** I requested a refund for purchases I mistakenly made on the App Store and requested refunds, it’s past 48 hours, but nothing.
- **Reply:** I’m sorry you’re still waiting on the refund. Refunds can take up to 7 business days to appear, so please check the Purchase History on your device or in the App Store app for the status. If it hasn’t updated after that period, let me know and we can review the request further.
- **Top retrieved:** sim=0.26 | cust=`I can’t download apps from the app store.` | agent=`Let’s work together to get your apps downloading again. Are you downloading updates or new apps? Any messages when downl`
- **Heuristic why:** Token overlap with retrieved brand replies = 0; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A5. id=1645444 · groundedness=3 · judge_mean=3.67 · top_sim=0.17

- **Message:** - I’m trying to backup my photos from my Macbook to my iCloud account. Been downloading for more than 24 hours. Sound correct?
- **Reply:** It sounds like the transfer is taking longer than usual—large libraries or a slow connection can cause this. Try pausing the upload, closing the Photos app, and then reopening it to see if progress resumes. If it still stalls, check your Wi‑Fi speed or switch to a wired connectio
- **Top retrieved:** sim=0.17 | cust=`So randomly deleted over 3,000 photos from my phone. Wow wtf!!` | agent=`We know how precious your memories are, and we'll help any way we can. Which iOS version is running on your iPhone? Do y`
- **Heuristic why:** Token overlap with retrieved brand replies = 1; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A6. id=1474846 · groundedness=3 · judge_mean=3.67 · top_sim=0.19

- **Message:** hi, I had money removed from my account only for my payment method to be declined later and no purchase. Refund please.
- **Reply:** We’re sorry for the inconvenience.  
We’ll review the charge that was taken before the decline and initiate a refund.  
Please send us your Apple ID or the email you use for Apple services so we can assist you further.
- **Top retrieved:** sim=0.19 | cust=`Help!! I need to update the payment method for my iPhone X pre-order. It won’t let me do it online` | agent=`We're happy to help you with that. DM us the country you are located in so we can better assist you:`
- **Heuristic why:** Token overlap with retrieved brand replies = 2; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A7. id=301581 · groundedness=3 · judge_mean=3.67 · top_sim=0.39

- **Message:** não faz nem 1 ano q comprei o IPhone 7 e ele já desliga com 50% de bateria, q bela bosta cara vcs fizeram eim !
- **Reply:** Entendo como isso pode ser frustrante. O primeiro passo é verificar a saúde da bateria no seu iPhone nas configurações; se a capacidade máxima estiver baixa, podemos agendar uma troca. Caso precise de ajuda imediata, a equipe de suporte técnico pode orientar um reset ou encaminha
- **Top retrieved:** sim=0.39 | cust=`Não melhorou. PIOROU! Sai de casa com 100% de bateria e cheguei no trampo com 92%... SEM USAR! please, fiX the battery i` | agent=`Since our Twitter support is available in English, get help at or join`
- **Heuristic why:** Token overlap with retrieved brand replies = 0; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A8. id=2051133 · groundedness=3 · judge_mean=4.00 · top_sim=0.25

- **Message:** iOS 11 sucks can I go back to 10?
- **Reply:** I’m sorry to hear how frustrating iOS 11 is for you.  
Only devices that Apple is still signing can be downgraded, and iOS 10 is no longer signed for most models.  
If you have a device that still accepts iOS 10, we can walk you through restoring it via iTunes or Finder.  
Otherw
- **Top retrieved:** sim=0.25 | cust=`Hey iOS 11 sucks none of my apps can update anymore` | agent=`Let us know via DM what occurs when trying to update, then we can work together to get it resolved.`
- **Heuristic why:** Token overlap with retrieved brand replies = 1; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A9. id=242547 · groundedness=3 · judge_mean=4.00 · top_sim=0.24

- **Message:** It’s like I’m only pretend controlling it because Apple thinks I’m too dumb to know when I want WiFi. Same thing w. screen brightness.
- **Reply:** I’m sorry this is frustrating.  
To control Wi‑Fi manually, open **Settings → Wi‑Fi** and toggle off “
- **Top retrieved:** sim=0.24 | cust=`u too` | agent=`We are here for you, What apps are you having issues with?`
- **Heuristic why:** Token overlap with retrieved brand replies = 0; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A10. id=2375264 · groundedness=3 · judge_mean=4.00 · top_sim=0.23

- **Message:** since updating my iPhone 6 to the latest iOS, my screen is just freezing up and becoming non-responsive. help! I'm working from Indonesia right now and not sure how to get support
- **Reply:** It sounds like a software hiccup. First try a hard‑restart: hold the Home button and the Sleep/Wake button together until the Apple logo appears.  
If the screen still freezes, go to **Settings → General → Software Update** and install any available updates, then check for proble
- **Top retrieved:** sim=0.23 | cust=`Ey since I updating my iPhone 8 to iOS 11.1 does not stop restarting!! 😡😞😭 #iOS11 #bug` | agent=`Let's get this sorted out together. Do you notice this happening when performing any specific steps or opening an app?`
- **Heuristic why:** Token overlap with retrieved brand replies = 1; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A11. id=2713217 · groundedness=3 · judge_mean=4.00 · top_sim=0.20

- **Message:** Hey you should make it so you have to put in a passcode to turn the phone off and please take airplane mode/cellular off the locked control center. Would make it a little harder for thieves to get away with stolen phones 🙃 #byebye1000dollar
- **Reply:** I’m glad you shared your idea—Apple does hear a lot of feedback from users like you.  
At this time there isn’t a setting to require a passcode to power off the device or to remove airplane mode from the locked Control Center.  
If you’d like to make the phone less vulnerable, yo
- **Top retrieved:** sim=0.20 | cust=`Ios 11.1 & I can’t even turn off the wifi & bluetooth from the control center!!` | agent=`We can help. Here's a great article to clarify more about Wi-Fi & Bluetooth in the Control Center:`
- **Heuristic why:** Token overlap with retrieved brand replies = 1; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A12. id=878559 · groundedness=3 · judge_mean=4.00 · top_sim=0.31

- **Message:** Need help in cancelling a subscription
- **Reply:** You can cancel the subscription directly from your device. Open Settings, tap your name → Subscriptions, find the app, and select “Cancel Subscription.” If you’re using a Mac, go to the App Store, click your account, and manage your subscriptions from there. Let us know if you ru
- **Top retrieved:** sim=0.31 | cust=`help?` | agent=`We're here to help in any way we can. Since we offer support in English, could you let us know what the message says?`
- **Heuristic why:** Token overlap with retrieved brand replies = 0; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A13. id=845901 · groundedness=3 · judge_mean=4.00 · top_sim=0.19

- **Message:** Does anyone know why my phone doesn’t backup to iCloud? Although I have 50G space! #Help #iPhone
- **Reply:** Make sure iCloud Backup is enabled: Settings > [Your Name] > iCloud > iCloud Backup, then tap “Back Up Now.”  
Verify you’re connected to Wi‑Fi and that the battery isn’t too low.  
If the backup still fails, toggle iCloud Backup off and on again and try a new backup.
- **Top retrieved:** sim=0.19 | cust=`Anyone know how to fix this? cc/ #AppleWatch` | agent=`We're here to help. Give these steps a try: If you still have issues, DM us.`
- **Heuristic why:** Token overlap with retrieved brand replies = 2; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A14. id=1418031 · groundedness=3 · judge_mean=4.00 · top_sim=0.25

- **Message:** 4.0.1 yes I restarted it
- **Reply:** Thanks for letting us know you restarted. Can you describe what happens when you power it on now? If the screen stays on the Apple logo, we can try a soft reset or check for updates. Let us know the exact behavior so we can guide you further.
- **Top retrieved:** sim=0.25 | cust=`my phone was acting up. I then restarted it and now all it does is stay on the apple logo when I turn it on. Wtf` | agent=`We’ve got your back. Let us know if the steps in this article help with your iPhone:`
- **Heuristic why:** Token overlap with retrieved brand replies = 1; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

### A15. id=1535710 · groundedness=3 · judge_mean=4.00 · top_sim=0.21

- **Message:** how do u cancel subscriptions that come out every month x
- **Reply:** To stop a recurring subscription, open **Settings** on your iPhone or iPad, tap your name at the top, then select **Subscriptions**.  
Find the subscription you want to cancel, tap it, and choose **Cancel Subscription**.  
If it’s on a Mac, go to **Apple menu > System Settings > 
- **Top retrieved:** sim=0.21 | cust=`HOW DO I️ FIX THIS SOMEONE HELP` | agent=`Let's dive into this issue together. Which iOS version is your device currently running?`
- **Heuristic why:** Token overlap with retrieved brand replies = 0; heuristic maps that to groundedness 3 (3 + min(2, overlap//3)).

## Similarity vs ablation

- High sim (≥0.35): n=6, mean Δ groundedness=+0.83
- Low sim (<0.35): n=34, mean Δ groundedness=+0.29
- Rows where retrieval helped (Δ>0): 14/40; mean top_sim among helped=0.36 (overall mean top_sim=0.29)

## Track B — routing / escalation (n=200)

- False negatives (should escalate, did not): **0** — prioritize these
- False positives (escalated, gold says no): **103** (51.5%); auto-handle rate=10.5% (first 30 FPs saved in JSON)

### FP top_sim distribution

- median=0.207; share &lt;0.25=74.8%; share in [0.30,0.35)=6.8%
- [0.00,0.10): 0 (0.0%)
- [0.10,0.15): 11 (10.7%)
- [0.15,0.20): 33 (32.0%)
- [0.20,0.25): 33 (32.0%)
- [0.25,0.30): 19 (18.4%)
- [0.30,0.35): 7 (6.8%)

Interpretation: if near-threshold share is tiny and most FPs sit well
below 0.25, this is retrieval **coverage**, not a 0.35 nudge.

### False negatives

None on this golden set (recall=100%). Trust risk is over-escalation,
not missed handoff — still call that out as a precision tradeoff.

### False positive sample (first 10)

- id=1648364 top_sim=0.13 reason=`No strong similar precedent (top similarity 0.13 < 0.35).` | msg=`Minha esposa comprou o celular iPhone 7 a 6 meses. Vive travando. E o suporte da Apple não resolve! Vergonha. !`
- id=1718433 top_sim=0.16 reason=`No strong similar precedent (top similarity 0.16 < 0.35).` | msg=`I’ve tried restarting it multiple times but it still won’t let me and I only have one user account`
- id=764855 top_sim=0.17 reason=`No strong similar precedent (top similarity 0.17 < 0.35).` | msg=`my mini won’t boot, rebooted for the recent high sierra patches and now stuck in failing boot loop re: OSInstall.mpkg`
- id=2095339 top_sim=0.16 reason=`No strong similar precedent (top similarity 0.16 < 0.35).` | msg=`iOS 11 is buggy bros the patch needs another patch. System crashes. Total 🛑 then buffering then it reboots like a good soldier.`
- id=1817632 top_sim=0.24 reason=`No strong similar precedent (top similarity 0.24 < 0.35).` | msg=`my son picked up his phone and this is on the screen?? HELP`
- id=769635 top_sim=0.28 reason=`No strong similar precedent (top similarity 0.28 < 0.35).` | msg=`My mail app disappeared,had to delete and reinstall,it’s still not working ok,wifi keeping coming on its own,volume buttons were not working`
- id=1541125 top_sim=0.19 reason=`No strong similar precedent (top similarity 0.19 < 0.35).` | msg=`I can’t transfer my data from old iPhone 7 to new iPhone X because IOS 11.2 isn’t available for the X. What can I do?`
- id=1386749 top_sim=0.19 reason=`No strong similar precedent (top similarity 0.19 < 0.35).` | msg=`worst update ever, ios11😡😡😡😡 its effing awful, drains battery, horrible layout, choppy, who came up with this??`
- id=807402 top_sim=0.25 reason=`No strong similar precedent (top similarity 0.25 < 0.35).` | msg=`WiFi... cheers`
- id=2955461 top_sim=0.24 reason=`No strong similar precedent (top similarity 0.24 < 0.35).` | msg=`why doing ave to restart my phone twice a day? Frozen screen. iMessage inaccessible. What happened to quality? Gonna Ditch iPhone.`

## Reproduce

```bash
PYTHONPATH=. python eval/run_eval.py --draft llm --reply-sample 40 --workers 3
PYTHONPATH=. python scripts/extract_failures.py
```
