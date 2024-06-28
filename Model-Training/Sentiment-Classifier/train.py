import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


emotion_mapping_score = {
    0: 1.0,   # anger
    1: 0.5,   # disgust
    2: 0.8,   # fear
    3: -1.0,  # happiness
    4: 1.0,   # sadness
    5: -0.8,  # surprise
    6: 0.0    # neutral
}

def train_ill(model, optimizer, train_loader, device, criterion_emotion, criterion_illness):
    model.train()
    total_loss_emotion = 0.0
    total_loss_illness = 0.0


    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        emotion_probs, illness_score = model(data)
        

        # Calculate losses
        loss_emotion = criterion_emotion(emotion_probs, data.y)
        emotion_labels = torch.tensor([emotion_mapping_score[label.item()] for label in data.y], device=device)
        loss_illness = criterion_illness(illness_score, emotion_labels.float())  


        # Accumulate total losses
        total_loss_emotion += loss_emotion.item()
        total_loss_illness += loss_illness.item()


        # Backpropagation and optimization
        loss_emotion.backward(retain_graph=True)
        loss_illness.backward()
        optimizer.step()


    # Compute average losses
    avg_loss_emotion = total_loss_emotion / len(train_loader)
    avg_loss_illness = total_loss_illness / len(train_loader)

    return avg_loss_emotion, avg_loss_illness



def test(model, test_loader, device):
    model.eval()
    correct_emotion = 0
    total_emotion = 0
    total_loss_illness = 0.0
    

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            emotion_probs, illness_score = model(data)
            

            # Calculate emotion predictions
            _, predicted_emotion = torch.max(emotion_probs, 1)
            correct_emotion += (predicted_emotion == data.y).sum().item()
            total_emotion += data.y.size(0)


            # Calculate illness loss
            emotion_labels = torch.tensor([emotion_mapping_score[label.item()] for label in data.y], device=device)
            criterion_illness = torch.nn.MSELoss()
            loss_illness = criterion_illness(illness_score, emotion_labels.float())


            # Accumulate total illness loss
            total_loss_illness += loss_illness.item()


    # Compute test accuracy
    test_accuracy = correct_emotion / total_emotion


    return test_accuracy, total_loss_illness



def test_first_10(model, test_loader, device):
    model.eval()
    count = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            emotion_probs, illness_score = model(data)


            # Convert emotion_probs and illness_score to CPU for printing if necessary
            emotion_probs = emotion_probs.cpu().numpy()
            illness_score = illness_score.cpu().numpy()


            # Print or store individual outputs for the first 10 samples
            for i in range(len(data)):
                if count >= 10:
                    break
                
                print(f'Data Point {count + 1}:')
                print(f'Emotion Probabilities: {emotion_probs[i]}')
                print(f'Illness Score: {illness_score[i]}')

                count += 1


            if count >= 10:
                break


def evaluate(model, val_loader, device, criterion_emotion, criterion_illness):
    model.eval()
    total_loss_emotion = 0.0
    total_loss_illness = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            emotion_probs, illness_score = model(data)

            # Calculate losses
            loss_emotion = criterion_emotion(emotion_probs, data.y)
            emotion_labels = torch.tensor([emotion_mapping_score[label.item()] for label in data.y], device=device)
            loss_illness = criterion_illness(illness_score, emotion_labels.float())

            # Accumulate total losses
            total_loss_emotion += loss_emotion.item()
            total_loss_illness += loss_illness.item()

            # Store predictions and labels for metric computation
            preds = torch.argmax(emotion_probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    # Compute average losses
    avg_loss_emotion = total_loss_emotion / len(val_loader)
    avg_loss_illness = total_loss_illness / len(val_loader)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss_emotion, avg_loss_illness, accuracy, precision, recall, f1

