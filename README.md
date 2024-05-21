# Название проекта

## О проекте
Это приложение предназначено для диагностики и распознавания состояния растений по изображениям листьев. Оно использует сверточные нейронные сети для анализа фотографий, предоставляя пользователям быструю обратную связь о здоровье растений.

## Архитектура приложения

### Backend
Backend приложения разработан на Python с использованием фреймворка Flask и развернут на Google Cloud. Он отвечает за обработку запросов от frontend, выполнение моделей машинного обучения для анализа изображений и взаимодействие с базой данных Firebase для хранения результатов.

#### Основные компоненты:
- **Flask**: Обеспечивает создание и обработку веб-запросов.
- **Google Cloud**: Используется для хостинга сервера и обеспечения масштабируемости.
- **Firebase**: Используется для хранения изображений и результатов анализа.

### Frontend
Frontend разработан на Kotlin и использует Firebase для управления данными пользователя и хранения изображений. 

#### Основные компоненты:
- **Kotlin**: Язык программирования для разработки Android приложения.
- **Firebase Storage**: Хранение изображений, загруженных пользователями.

## Установка

### Установка APK
Для установки приложения на Android устройство следуйте следующим шагам:

1. **Скачайте APK файл**
   - Скачайте файл `.apk` из предоставленной ссылки или репозитория.

2. **Разрешите установку из неизвестных источников**
   - Перейдите в настройки вашего устройства -> Безопасность -> и включите опцию "Неизвестные источники".

3. **Установите APK**
   - Откройте скачанный файл на вашем устройстве и следуйте инструкциям для установки.

4. **Запуск приложения**
   - После установки вы найдете иконку приложения в меню. Тапните по ней для запуска.

