class NN(nn.Module):

    def __init__(self, num_input, hidden_units, num_output, device=torch.device('cpu')):
        super().__init__()
        self.num_sensors = num_input  # this is the number of features
        self.hidden_units = hidden_units
        self.model = nn.Sequential(
            nn.Linear(num_input, hidden_units),
            nn.Sigmoid(),
            nn.Linear(hidden_units, num_output),
            nn.Sigmoid(),
        )
        self.device = device
        self.to(device=device)

    def forward(self, input):
        out = self.model(input)
        return out


class Scaler1(object):
    # Temp type 2
    # MW as input, type 2
    # MW as output, type 1

    def __init__(self, Min, Max, kind:int):
        self.Min = Min
        self.Max = Max
        self.kind = kind
        self.f1 = np.sqrt(2.0 * Max)
        self.f2 = np.sqrt(0.5 * Min)

    def check(self):
        x = np.random.uniform(self.Min, self.Max, size=20)

        y = self.type_0(x)
        x0 = self.inverse_type_0(y)
        assert np.sum(np.abs(x-x0)) < 1E-6

        y = self.type_2(x)
        x0 = self.inverse_type_2(y)
        assert np.sum(np.abs(x-x0)) < 1E-6

        y = self.type_3(x)
        x0 = self.inverse_type_3(y)
        assert np.sum(np.abs(x-x0)) < 1E-6

        self.Min = 0
        self.f2 = np.sqrt(0.5 * 0)
        x = np.random.uniform(0, self.Max, size=20)
        y = self.type_1(x)
        x0 = self.inverse_type_1(y)
        assert np.sum(np.abs(x-x0)) < 1E-6

        print("Scaler check pass")

    def transform(self, x):
        if self.kind == 0:
            return self.type_0(x)
        elif self.kind == 1:
            return self.type_1(x)
        elif self.kind == 2:
            return self.type_2(x)
        elif self.kind == 3:
            return self.type_3(x)
        else:
            raise Exception(f"Unimplemented type {self.kind}")

    def inverse_transform(self, x):
        if self.kind == 0:
            return self.inverse_type_0(x)
        elif self.kind == 1:
            return self.inverse_type_1(x)
        elif self.kind == 2:
            return self.inverse_type_2(x)
        elif self.kind == 3:
            return self.inverse_type_3(x)
        else:
            raise Exception(f"Unimplemented type {self.kind}")

    def type_0(self, x):
        return x

    def inverse_type_0(self, x):
        return x

    def type_1(self, x):
        y = (np.sqrt(x) - self.f2) / (self.f1 - self.f2)
        return y

    def inverse_type_1(self, x):
        y = x * (self.f1 - self.f2) + self.f2
        y = y * y
        return y

    def type_2(self, x):
        y=(0.8-0.2)*(x-self.Min)/(self.Max-self.Min)+0.2
        return y

    def inverse_type_2(self, x):
        y = (self.Max - self.Min) * (x - 0.2) / (0.8 - 0.2) + self.Min;
        return y

    def type_3(self, x):
        y = (5.0 - 0.0) * (x - self.Min) / (self.Max - self.Min) + 0.0;
        return y

    def inverse_type_3(self, x):
        y = (self.Max - self.Min) * (x - 0.0) / (5.0 - 0.0) + self.Min;
        return y


class Scaler(object):

    def __init__(self, Min, Max, type=1):
        self.Min = Min
        self.Max = Max
        self.f1 = np.sqrt(2.0 * Max)
        self.f2 = np.sqrt(0.5 * Min)

    def transform(self, y):
        return (y.pow(0.5) - self.f2)/(self.f1-self.f2)

    def inverse_transform(self, y):
        return (y * (self.f1 - self.f2) + self.f2).square()


def standardize(self, df_train, df_test):
        stdscaler = StandardScaler()
        stdscaler.fit(df_train)
        Min = df_train[self.target].values.flatten().min()
        Max = df_train[self.target].values.flatten().max()
        scaler = Scaler1(Min=Min, Max=Max, kind=1)
        self.stdscaler = stdscaler
        self.scaler = scaler

        df_train_s = pd.DataFrame(self.stdscaler.transform(df_train), columns=df_train.columns, index=df_train.index)
        df_test_s = pd.DataFrame(self.stdscaler.transform(df_test), columns=df_val.columns, index=df_test.index)

        # normalize mvs
        df_train_s[self.target] = mwscaler.transform(df_train[self.target])
        df_test_s[self.target] = mwscaler.transform(df_test[self.target])
        return df_train_s, df_test_s




DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

train_X = torch.tensor(df_train_s[features].values).float()
train_Y = torch.tensor(df_train_s[target].values).float()
train_Y0 = torch.tensor(df_train[target].values).float()
test_X = torch.tensor(df_test_s[features].values, device=DEVICE).float()
test_Y = torch.tensor(df_test_s[target].values, device=DEVICE).float()
test_Y0 = torch.tensor(df_test[target].values, device=DEVICE).float()


model = NN(num_input=len(features), hidden_units=25, num_output=12, device=DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
loss_function = nn.MSELoss(reduction='sum')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.9)

num_train = train_Y.nelement()
num_tests = test_Y.nelement()

res = []
model.train()
for ix_epoch in range(n_epochs):
    t1 = datetime.now()
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]

    total_loss = 0
    for X, y, y0 in train_loader:
        yhat = model(X)
        loss = loss_function(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y1 = self.mwscaler.inverse_transform(yhat)
            total_loss += loss_function(y1, y0).item()

    rmse_train_mw = np.sqrt(total_loss / num_train)
    with torch.no_grad():
        y1 = self.mwscaler.inverse_transform(model(val_X))
        total_loss = loss_function(y1, val_Y0).item()
        rmse_val_mw = np.sqrt(total_loss / num_tests)

        y1 = self.mwscaler.inverse_transform(model(test_X))
        total_loss = loss_function(y1, test_Y0).item()
        rmse_test_mw = np.sqrt(total_loss / num_tests)

    t2 = datetime.now()
    if (ix_epoch + 1) % 500 == 0:
        print(
            f"Epoch {ix_epoch + 1}/{n_epochs}  {(t2 - t1).seconds} seconds RMSE_train: {rmse_train_mw:.2f}  RMSE_val: {rmse_val_mw:.2f}  RMSE_test: {rmse_test_mw:.2f}  LR: {lr}")
        print('---------')

    res_ip = {'epoch': ix_epoch, 'RMSE_train': rmse_train_mw, 'RMSE_val': rmse_val_mw, 'RMSE_test': rmse_test_mw}
    res.append(res_ip)
