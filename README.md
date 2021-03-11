## Code for paper "Hiding in Plain Sight: Adversarial Attack via Style Transfer on border Borders"

### Requirements
* python 3.6+
* pytorch 1.3.1
* torchvision 0.4.2
* pillow 6.2.1
* numpy 1.17

### Organization<br>
<b>borderAttack:</b> targeted and untargeted adversarial border attack.<br>
<b>att-mid:</b> generalize the attack region and take the middle region as example.<br>

<b>Subfolders:</b><br>
  /sty_adv_border : implement the attack methods.<br>
  /dataset : store clean images and clean borders.<br>
  /results : store trained borders and generated adversarial borders.<br>
### Running
```
cd sty_adv_border
python main.py --target=1 //for targeted attack
python main.py --target=0 //for untargeted attack
```

### subject human image quality assessment
The website is [here](http://39.96.66.112/instructor2.html)
