# Orchestrator End-to-End Test Cases
 
 
## 1. Car Control Agent
1. Open the driver side window
2. Set the ambient lighting to polar blue
3. Turn on the seat heating for the passenger
4. Fold the side mirrors
5. Turn on the rear window defroster
6. Activate the seat massage function
7. Turn off the Burmester sound system
8. Lower the rear windows
9. Turn on the hazard lights
10. Open the glove compartment
11. Adjust the steering column down
12. Activate sport suspension mode
13. Unlock the passenger door
14. Turn on the high beams
15. Open the charging port door
16. Turn on the fog lights
17. Activate eco mode
18. Lower the sunblinds
 
## 2. Car Manual Agent
1. How do I pair my phone with the MBUX system?
2. What is the recommended tire pressure for driving on the Autobahn?
3. How to activate the Distronic active distance assist?
4. Where is the first aid kit located in the trunk?
5. How do I refill the AdBlue fluid?
6. What does the yellow engine warning light mean?
7. How to use the active parking assist feature?
8. Where can I find the fuse box?
9. How often does my C-Class need a service B?
10. How to change the windshield wiper blades?
11. How do I connect to the Mercedes me connect app?
12. What is the towing capacity of the GLE?
13. How to adjust the head-up display brightness?
14. How do I reset the trip computer?
15. Where is the manual release for the charging cable?
16. How to operate the Magic Body Control?
17. What type of engine oil should I use?
18. How to jump start the vehicle?
19. How to replace the key fob battery?
20. How to fold down the rear seats?
21. What is the fuel tank capacity of the E-Class?
22. How to use the augmented reality navigation?
23. How do I enable the child safety locks?
24. How to check the transmission fluid level?
25. Where is the OBD2 port located?
26. How to turn on the interior motion sensor?
27. How to use the Linguatronic voice control?
28. How to properly wash the matte paint finish?
29. How to calibrate the 360-degree camera?
30. What should I do before driving my Mercedes-Benz Actros, Antos, or Arocs?
31. What documents should be kept in my Mercedes-Benz vehicle?
32. How can I access the online Operating Instructions for my Mercedes-Benz truck?
33. What are some tips for environmentally responsible operation of my vehicle?
34. Where should maintenance work on my Mercedes-Benz vehicle be carried out?
35. What parts should I use for my Mercedes-Benz vehicle to ensure safety?
36. What happens if I connect equipment to the diagnostics connection in my vehicle?
37. What is required for the BlueTec exhaust gas aftertreatment system to function correctly?
38. What happens if I operate my vehicle without AdBlue in the BlueTec system?
39. How should I correctly fasten a seat belt in my Mercedes-Benz vehicle?
40. What should I do if the restraint system warning lamp lights up?
41. How should I secure a child in my Mercedes-Benz vehicle?
42. What should I do if I lose a key for my Mercedes-Benz vehicle?
43. How do I replace the battery in the remote control key?
44. What functions can I access through the vehicle check menu on the multifunction key?
45. How can I perform a lamp check using the multifunction key?
46. How do I lock or unlock the vehicle doors using the integrated key?
47. What is the convenience closing feature and how do I use it?
48. What does the ATA (anti-theft alarm system) monitor on my vehicle?
49. How do I prime the ATA system on my vehicle?
50. What happens when the ATA system triggers an alarm?
51. How should batteries be disposed of?
52. How do you replace the battery in the multifunction key?
53. What are the risks of leaving children unattended in the vehicle?
54. How do you open or close the side windows?
55. How do you reset the side windows after a malfunction?
56. How do you operate the roller sunblinds?
57. How should seats be adjusted for safety?
58. How do you adjust the exterior mirrors?
 
## 3. Infotainment Agent
1. Play some classical music by Beethoven
2. Tell me a joke about driving on the Autobahn
3. Play 99 Luftballons on YouTube
4. Find the latest news broadcast on the radio
5. Tell me a funny story (NOT good)
6. Play a podcast about Mercedes-Benz history
7. Play some German techno music
8. Put on a rock playlist
9. Tell me a knock-knock joke
10. Play the top hits in Berlin
11. I want to listen to some jazz
12. Tell me a joke about a mechanic
13. Play an audiobook
14. Put on some relaxing lounge music
15. Play electronic dance music
16. Give me a good joke to make me laugh
17. Play Mozart symphony number 9
18. Find a hip hop track on YouTube
19. Tell me a joke about cars
20. Play traditional Bavarian music
21. Play something upbeat for a road trip
22. Tell me a short joke
23. Play acoustic guitar music
24. Find the song Autobahn by Kraftwerk
25. Tell me a joke about traffic
26. Play some ambient background music
27. Play the latest pop songs
28. Make me laugh with a clever joke
29. Play an instrumental piano piece
30. Put on a heavy metal playlist
## 4. Navigation Agent
1. Navigate to the Mercedes-Benz Museum in Stuttgart
2. Are there any Aldi supermarkets around here?
3. Show me the way to the Cologne Cathedral
4. Are there any traffic jams on the A3 right now?
5. Direct me to the Neuschwanstein Castle
6. Where is the closest public restroom?
7. Navigate to Europa-Park
8. Find a rest stop on the highway
9. Route to the Black Forest
10. Navigate to Com Tam Cali for me please
11. Which place is closer: Pho 24 or Diamond Plaza?
12. Give me directions to Com Tam Cali from here.
 
## 5. Cloud Agent
1. Who won the latest Bundesliga football match?
2. What is the population of Berlin?
3. Summarize the history of Karl Benz
4. What are the top tourist attractions in Germany?
5. How tall is the TV Tower in Berlin?
6. Give me the latest news from the European tech industry
7. Who is the current Chancellor of Germany?
8. What is the capital of Bavaria?
9. How old is the Mercedes-Benz brand?
10. Tell me about the history of the Autobahn
11. What is the largest lake in Germany?
12. Give me the latest stock market updates in Frankfurt
13. Who was Albert Einstein?
14. How many car manufacturers are there in Germany?
15. What is the recipe for an authentic German pretzel?
16. Summarize the history of the Berlin Wall
17. What are the current top headlines in Europe?
18. Give me a brief overview of the European Union
19. What is the most popular sport in Germany?
20. Who is the CEO of Mercedes-Benz?

## 6. Multi-Action Compound Commands (control_car batch)
Mọi câu dưới đây PHẢI thực hiện ĐỦ các action, trả lời xác nhận đủ từng phần
(không drop action nào, không hỏi lại "which one").

| # | Câu lệnh | Kỳ vọng |
|---|---|---|
| 1 | Open left and right door | mở cả left_door + right_door |
| 2 | Open the trunk, and open left door | mở trunk + left_door |
| 3 | Turn on the light and the ac | bật light + ac |
| 4 | Open both doors | mở cả 2 cửa |
| 5 | Close all doors | đóng cả 2 cửa |
| 6 | Open left door and right door | mở cả 2 cửa |
| 7 | Open the trunk and close the left door | mở trunk + đóng left_door |
| 8 | Mở cửa trái và cửa phải | mở cả 2 cửa |
| 9 | Mở cốp và mở cửa trái | mở trunk + left_door |
| 10 | Bật đèn và điều hòa | bật light + ac |
| 11 | Mở cả hai cửa | mở cả 2 cửa |
| 12 | 左のドアと右のドアを開けて | mở cả 2 cửa (ja) |
| 13 | Open the trunk and turn on the light | mở trunk + bật light |

Regression (1 lệnh đơn phải chạy như cũ):
| # | Câu lệnh | Kỳ vọng |
|---|---|---|
| 14 | Open the left door | chỉ left_door |
| 15 | Turn off ac | chỉ ac |
| 16 | Đóng cửa phải | chỉ right_door |

Phần kết quả:
- Log `[tool_call] control_car actions=N commands=[...]` với N đúng số action.
- Xác nhận đọc đủ N câu (en/vi/ja theo ngôn ngữ nói).
- Thất bại 1 phần (vd mất kết nối GraphQL 1 lệnh): "Xin lỗi, không thể thực hiện: ..." + các câu thành công vẫn đọc.

## 7. Voice Interruption (barge-in + wake-word khi đang chờ)
| # | Tình huống | Kỳ vọng |
|---|---|---|
| 1 | Đang chờ orchestrator trả lời → nói "Hey Dora" | ngắt turn cũ, session MỚI, không trộn context |
| 2 | Dora đang nói → nói "Hey Dora, stop" | TTS cắt ngay, turn mới bắt đầu |
| 3 | Dora đang nói → nói chèn (VAD barge-in) | TTS cắt, bắt turn mới (giữ session) |
| 4 | Ngắt giữa chừng rồi hỏi tiếp | không có audio cũ leak ra sau khi ngắt |
| 5 | "Open left and right door" rồi ngắt giữa câu trả lời | 2 cửa đã mở, câu trả lời bị bỏ dở không đọc tiếp |