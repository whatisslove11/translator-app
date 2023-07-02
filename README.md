<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">NEW-GEN TRANSLATOR</h3>

  <p align="center">
    <i>Репозиторий для выполнения проекта «Модель переводичка на английский» </i>
    <br />
    <i>для курса <strong>«Разработка ML сервиса: от идеи к прототипу»<strong></i>
    <br />
    <br />
    <!--добавить метки нужным полям-->
    <a href="https://github.com/whatisslove11/translator-app/tree/main/documentation"><mark>Изучите документацию здесь</mark></a> 
    <br />
    <a href="https://github.com/whatisslove11/hse-proj/issues">Сообщить об ошибке</a>
    ·
    <a href="https://github.com/whatisslove11/hse-proj/issues">Предложить улучшение</a>
  </p>
</div>


<!-- FILES -->
## Files

- `app.py` - streamlit app file
- `decode.py`: script for normalizing text and future translation
- `model.pt`: weights for Transformer model
- `requirements.txt`: package requirements files
- `en_dict.pkl` and `de_dict.pkl`: dictionaries for translation
- `model` dir: modules for Translation model

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>



<!-- ВОЗМОЖНЫЕ УЛУЧШЕНИЯ -->
## Возможные улучшения

<h4>Данная версия модели была обучена относительно давно — еще до того момента, как автор погрузился в мир возможных модификаций трансофрмеров — поэтому улучшений в эту модель не внесено никаких, а новая модель все еще стоит на обучении. Однако список того, что можно улучшить в модели, вы можете увидеть ниже: </h4>

- [ ] weight tying
- [ ] flash attention
- [ ] pre LN
- [ ] torch.jit.script + torch.compile
- [ ] cosine warmup
- [ ] byte-pair encoding
- [ ] weight sharing
- [ ] accumulation gradients
- [ ] label smoothing in train
- [ ] test variants of small init (another type for some layers)
- [ ] train on wmt19
- [ ] positional encoding (PE/rotary)
- [ ] add more layers in encoder-decoder
- [ ] increase hidden_dim from 512 to 1024
- [ ] rewrite to pytorch lightinig (ddp, mixed precision, etc.)
- [ ] beam search\topk\topp\temperature in generaion tokens (test all these types)

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>

<!-- ПРОБЛЕМЫ РЕАЛИЗАЦИИ -->
## Проблемы реализации

<h4>Если автор будет вспоминать все проблемы во время реализации моделей, он заплачет -(. </h4>

Базовая модель — модель без единого улучшения из пункта выше.\
Улучшенная модель — модель с большим количеством внедрений из пункта выше.\

Проблемы с разработкой базовой модели:
- очень долгое время подготовки датасета
- применение не самого удачного вида токенизации (из-за него датасет разрастался многократко, а также не был вариантивным - т.е. устойчивым к появлениям формы слова в переоде, а не конкретно этого слова)
- **очень** долгое время обучения
- много UNK токенов в итоговом переводе

Проблемы с разработкой улучшенной модели:
- проблемы с токенизацией (какой токенизатор взять, сколько токенов оставить, как удалить мусорные токеры, etc.)
- проблема "выкинутых букв" из токенизатора (моему токенизатору **очень** не нравились уникальные немецкие буквы по типу Ü, Ö, Ä и он их просто UNK токенами заменял)
- проблема с Flash Attention (спасибо PyTorch forums, что помогли)
- проблема с итоговым переводом - оно вообще не то переводило, поэтому ушло на переработку))

Однако преимущество улучшенной модели были видно уже во время обучения — в то время как базовой модели на 600.000 слов требовалось 120 минут на 1 эпоху, улучшенная модель тратит на 1.300.000 слов 71 минуту на 1 эпоху (и имеет на 30.000.000 параметров меньше)

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>
